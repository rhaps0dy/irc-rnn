from random import randint
import os
import re
import datetime
import unicodedata
import collections
import codecs
import numpy as np
import utils

class TextLoader():
    def __init__(self, save_dir, batch_size=20, file_size=20000, utterance_dependency_length=5):
        utils.makedirs_(save_dir)
        self.meta_file = os.path.join(save_dir, 'meta.pkl')
        self.parsed_dir = os.path.join(save_dir, 'parsed')
        self.samples_dir = os.path.join(save_dir, 'samples')

        self.batch_size = batch_size
        self.file_size = file_size
        self.utterance_dependency_length = utterance_dependency_length

        self.channel_last_files = collections.Counter()
        self.parsed_files = set()
        self.seq_length = 0

        # The vocabulary is ASCII 32 (' ') to 126 ('~'), plus special "end of X"
        # characters, plus others. The "unknown" character is meant to represent
        # any UTF-8 that is not just diacritics.
        self.chars = list(chr(i) for i in range(32, 127))
        self.chars += ["start-channel", "start-self-nick", "start-sayer-nick",
                       "utter", "action", "topic", "change-nick", "join",
                       "quit", "unknown", "end-event", "go", "pad",]
        self.vocab = dict(zip(self.chars, range(len(self.chars))))

        self.blacklisted_channels = []
        self.whitelisted_channels = []


    def load(self):
        if not os.path.exists(self.meta_file):
            raise ValueError("File does not exist: '%s'" % self.meta_file)
        self.seq_length, self.channel_last_files, self.parsed_files = utils.load(self.meta_file)


    def save(self):
        utils.dump((self.seq_length, self.channel_last_files, self.parsed_files), self.meta_file)

    def vocab_value(self, char):
        if char not in self.vocab:
            self.vocab[char] = self.vocab['unknown']
            try:
                uc = unicodedata.name(char)
                if ' WITH ' in uc:
                    c, _ = uc.split(' WITH ')
                    self.vocab[char] = self.vocab[unicodedata.lookup(c)]
                elif 'DASH' in uc:
                    self.vocab[char] = self.vocab['-']
            except ValueError:
                pass
            except KeyError:
                pass
        return self.vocab[char]


    def parse_data(self, data_dir):
        exp_start = r'^\[([0-9]{2}:[0-9]{2}:[0-9]{2})\] '
        exp_utterance = re.compile(exp_start + r'<([^ ]+)> (.*)$')
        exp_action = re.compile(exp_start + r'\* ([^ ]+) (.*)$')
        exp_topic = re.compile(exp_start + r"\*\*\* ([^ ]+) changes topic to '(.*)'$")
        exp_quit_join = re.compile(exp_start + r'\*\*\* (Quit|Join)s: ([^ ]+) .*$')
        exp_nick_change = re.compile(exp_start + r'\*\*\* ([^ ]+) is now known as ([^ ]+)$')
        # We're not interested in PM logs
        exp_log_file = re.compile(r'^(.*)_(#.*)_([0-9]{8}).log$')
        channels = {}

        def get_datetime(date, time):
            return datetime.datetime(year=int(date[0:4]),
                                     month=int(date[4:6]),
                                     day=int(date[6:8]),
                                     hour=int(time[0:2]),
                                     minute=int(time[3:5]),
                                     second=int(time[6:8]))
        def clean(s):
            return bytes(map(self.vocab_value, s))

        def new_channel():
            return {'nicks': collections.Counter(),
                    'events': [],
                    'max_utter_len': 0}

        def flush_channel(channel):
            channel_dir = os.path.join(self.parsed_dir, channel)
            utils.makedirs_(channel_dir)
            fpath = utils.pkl_f_n(channel_dir, self.channel_last_files[channel])
            self.channel_last_files[channel] += 1
            utils.dump(channels[channel], fpath)
            print("Wrote file %s" % fpath)
            del channels[channel]
            channels[channel] = new_channel()

        for fname in os.listdir(data_dir):
            if fname in self.parsed_files:
                continue
            m = exp_log_file.match(fname)
            if m is None:
                continue
            _network, channel, day = m.groups()
            if channel in self.blacklisted_channels or \
                (len(self.whitelisted_channels) > 0 and
                 channel not in self.whitelisted_channels):
                continue

            self.parsed_files.add(fname)
            with codecs.open(os.path.join(data_dir, fname), 'r', 'utf-8', 'ignore') as f:
                data = f.read().strip().split('\n')

            if channel not in channels:
                channels[channel] = new_channel()

            for line in data:
                m_list = [exp_utterance.match(line),
                          exp_action.match(line),
                          exp_topic.match(line),
                          exp_quit_join.match(line),
                          exp_nick_change.match(line)]

                m = None
                for e in m_list:
                    m = m or e
                if m is None:
                    # line not recognized
                    continue
                m = m.groups()

                if m_list[0] is not None or m_list[1] is not None or m_list[2] is not None:
                    nick = clean(m[1])
                    msg = clean(m[2])
                    nicks = [nick]
                    if m_list[0] is not None:
                        event = "utter"
                    elif m_list[1] is not None:
                        event = "action"
                    else:
                        event = "topic"
                    elem = (event, nick, msg)
                elif m_list[3] is not None:
                    nick = clean(m[2])
                    nicks = [nick]
                    # u'Quit' -> 'quit'
                    elem = (str(m[1]).lower(), nick)
                elif m_list[4] is not None:
                    nick_a = clean(m[1])
                    nick_b = clean(m[2])
                    nicks = [nick_a, nick_b]
                    elem = ("nick", nick_a, nick_b)

                dt = get_datetime(day, m[0])
                for nick in nicks:
                    channels[channel]['nicks'][nick] += 1
                channels[channel]['events'].append((dt, elem))
                if len(channels[channel]['events']) >= self.file_size:
                    flush_channel(channel)
        for c in channels:
            flush_channel(c)
        self.save()
        return channels


    def create_samples(self):
        def channel_self_to_input(channel, self_nick):
            l = []
            l.append(self.vocab["start-channel"])
            l += map(self.vocab_value, channel)
            l.append(self.vocab["start-self-nick"])
            l += map(self.vocab_value, self_nick)
            return l

        def event_to_input(time_event):
            _time = time_event[0]
            event = time_event[1]
            l = []
            l.append(self.vocab["start-sayer-nick"])
            l += list(event[1])
            if event[0] in ["join", "quit"]:
                l += [self.vocab[event[0]], self.vocab['end-event']]
                return l
            if event[0] in ["utter", "action", "topic"]:
                l.append(self.vocab[event[0]])
            else:
                l.append(self.vocab["change-nick"])
            l += list(event[2])
            l.append(self.vocab["end-event"])
            return l

        n = 0

        for channel in self.channel_last_files:
            print(channel)
            inp_d = os.path.join(self.parsed_dir, channel)
            tmp_d = os.path.join(self.parsed_dir, "tmp_%s" % channel)
            sorted_d = os.path.join(self.parsed_dir, "sorted_%s" % channel)
            out_d = os.path.join(self.samples_dir, channel)
            utils.makedirs_(tmp_d)
            utils.makedirs_(sorted_d)
            utils.makedirs_(out_d)

            channel_tag = channel_self_to_input(channel, "rhaps0dy")
            channel_lengths = []

            print("Creating sequences...")
            q = utils.CircularBufferQueue(max_elements=4)
            q.put(event_to_input((None, ("join", map(self.vocab_value, "rhaps0dy")))))
            for i in range(self.channel_last_files[channel]):
                chan_file = utils.load_f_n(inp_d, i)
                tmp_sequences = []
                for time_event in chan_file["events"]:
                    decoder = sum(q, channel_tag) # channel_tag + q[0] + q[1] ...
                    # Reverse input for increased performance
                    decoder.reverse()
                    e = event_to_input(time_event)
                    encoder = [self.vocab["go"]] + e
                    channel_lengths.append((len(decoder), len(encoder),
                                            len(channel_lengths)))
                    tmp_sequences.append((decoder, encoder))
                    if q.full():
                        q.get()
                    q.put(e)
                utils.dump_f_n(tmp_sequences, tmp_d, i)

            print("Sorting by length...")
            utils.external_sort(tmp_d, sorted_d, self.channel_last_files[channel], key=lambda t: (len(t[0]), len(t[1])))

            print("Padding, creating minibatches...")

            class OutTmp:
                out_tmp = []
                bsz = 0
                max_len_d = 0
                max_len_e = 0
                out = []
                def __init__(self, vocab):
                    self.vocab = vocab

                def flush_out_tmp(self):
                    l = len(self.out_tmp)
                    d_arr = [None]*l
                    e_arr = [None]*l
                    for j in range(l):
                        pad_d = [self.vocab["pad"]]*(self.max_len_d - len(self.out_tmp[j][0]))
                        pad_e = [self.vocab["pad"]]*(self.max_len_e - len(self.out_tmp[j][1]))
                        d_arr[j] = pad_d + self.out_tmp[j][0]
                        e_arr[j] = self.out_tmp[j][1] + pad_e
                        assert(len(d_arr[j]) == self.max_len_d)
                        assert(len(e_arr[j]) == self.max_len_e)

                    self.out.append((np.array(d_arr), np.array(e_arr)))

                    self.out_tmp = []
                    self.bsz = 0
                    self.max_len_d = 0
                    self.max_len_e = 0

            o = OutTmp(self.vocab)

            for i in range(self.channel_last_files[channel]):
                if len(o.out) != 0:
                    utils.dump_f_n(o.out, out_d, i-1)
                o.out = []

                events = utils.load_f_n(sorted_d, i)
                for event in events:
                    o.out_tmp.append(event)
                    o.max_len_d = max(o.max_len_d, len(event[0]))
                    o.max_len_e = max(o.max_len_e, len(event[1]))
                    o.bsz += 1
                    if o.bsz == self.batch_size:
                        o.flush_out_tmp()

            o.flush_out_tmp()
            i = self.channel_last_files[channel]-1
            utils.dump_f_n(o.out, out_d, i)
