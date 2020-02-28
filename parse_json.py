import json
import gzip
from rusenttokenize import ru_sent_tokenize
from nltk.tokenize import wordpunct_tokenize
import os

def load_gz_lines(path, encoding='utf8', gzip=gzip):
    with gzip.open(path) as file:
        for line in file:
            yield json.loads(line.decode(encoding).rstrip())


def parse_json_line(path_nerus):

    for indx, text in enumerate(load_gz_lines(path_nerus)):

        left_context_len, right_context_len = -1, -1

        for sentence in ru_sent_tokenize(text['content']):

            left_context_len, right_context_len = right_context_len+1, (right_context_len + len(sentence))+1
            annotation = (sentence, [])

            for indx, span in enumerate(text['annotations']):
                st_en = span['span']

                if (st_en['start']) >= left_context_len and (st_en['end']) <= right_context_len:
                    span = text['annotations'][indx]
                    span['span']['start'] -= left_context_len
                    span['span']['end'] -= left_context_len
                    annotation[1].append(span)

            start = 0
            labels = []
            words = []

            for ann in annotation[1]:

                piece = sentence[start:ann['span']['start']]
                ws = wordpunct_tokenize(piece)
                words += ws + [ann['text']]
                labels += (['O'] * len(ws)) + [ann['type']]
                start = ann['span']['end']

            piece = sentence[start:]
            ws = wordpunct_tokenize(piece)
            words += ws
            labels += (['O'] * len(ws))

            assert len(labels) == len(words), 'Mismatch len labels ans len sentence'
            #file = open(res_file, 'w+')
           #for word, tag in zip()

        break
    #print(sentences)

#parse_json_line('nerus_lenta.jsonl.gz')


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(file_path, mode='new'):
    #file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split(" ")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples

print(read_examples_from_file('test.txt'))

