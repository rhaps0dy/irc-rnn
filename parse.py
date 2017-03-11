#!/usr/bin/env python3

from random import randint
import os
import sys
import re
import datetime
import unicodedata
import collections
import codecs
import numpy as np
import utils
import pickle

class TextLoader():
    def __init__(self, save_dir, batch_size=64, file_size=2000000, utterance_dependency_length=4):
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
        # long-delay: the time since last event is more than 1 day
        # medium-delay: the time since last event is between 2h and 1 day
        # short-delay: the time since last event is between 2 min and 2h
        self.chars += ["start-sayer-nick", "short-delay", "medium-delay", "long-delay",
                       "utter", "action", "topic", "change-nick", "join",
                       "quit", "unknown", "end-event", "pad",]
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.reverse_vocab = [None]*len(self.vocab)
        for k, c in self.vocab.items():
            self.reverse_vocab[c] = k

        # Aliases
        self.vocab['START'] = self.vocab['start-sayer-nick']
        self.vocab['UNK'] = self.vocab['unknown']
        self.n_chars = len(self.chars)

        self.blacklisted_channels = ['#mpdm-rhaps0dy--tomaff--nish81-1', '#building-rukea']
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
            if len(channels[c]['events']) > 0:
                flush_channel(c)
        self.save()
        return channels


    def create_samples(self):
        def channel_to_input(channel):
            l = []
            l.append(self.vocab["start-channel"])
            l += list(map(self.vocab_value, channel))
            return l

        def event_to_input(event):
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

        all_sequences = []
        print("Creating sequences...")
        for channel in self.channel_last_files:
            print(channel)
            inp_d = os.path.join(self.parsed_dir, channel)
            out_f = os.path.join(self.parsed_dir, "sequences.pkl")

            for i in range(self.channel_last_files[channel]):
                chan_file = utils.load_f_n(inp_d, i)
                channel_sequences = []
                prev_time = datetime.datetime.fromtimestamp(0)
                prev_event = (None,)
                for time, event in chan_file["events"]:
                    if event[0] in ["join", "quit"] and prev_event[0] in ["join", "quit"]:
                        # Prevent quit/join spam
                        continue
                    prev_event = event
                    minutes_passed = (time-prev_time).total_seconds() / 60
                    prev_time = time
                    if minutes_passed > 1440:
                        channel_sequences.append([self.vocab["long-delay"]])
                        cur_seq = channel_sequences[-1]
                    elif minutes_passed > 120:
                        cur_seq.append(self.vocab["medium-delay"])
                    elif minutes_passed > 2:
                        cur_seq.append(self.vocab["short-delay"])
                    cur_seq += event_to_input(event)
                all_sequences += channel_sequences
        for i, seq in enumerate(all_sequences):
            all_sequences[i] = np.array(seq, dtype=np.int32)
        with open(out_f, 'wb') as f:
            pickle.dump(all_sequences, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: {:s} <ZNC_logs_dir> <save_dir>".format(sys.argv[0]))
        sys.exit(-1)
    tl = TextLoader(sys.argv[2])
    tl.parse_data(sys.argv[1])
    tl.create_samples()
