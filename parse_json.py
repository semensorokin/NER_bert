import json
import gzip
from rusenttokenize import ru_sent_tokenize
from nltk.tokenize import wordpunct_tokenize
from tqdm import tqdm

def load_gz_lines(path, encoding='utf8', gzip=gzip):
    with gzip.open(path) as file:
        for line in file:
            yield json.loads(line.decode(encoding).rstrip())


def parse_json_line(path_nerus, n = 30000 , save_to = 'train_nerus.txt'):

    flag = 0
    file = open(save_to, 'w+')

    for indx, text in tqdm(enumerate(load_gz_lines(path_nerus))):


        left_context_len, right_context_len = -1, -1
        flag += 1
        for sentence in ru_sent_tokenize(text['content']):

            left_context_len, right_context_len = right_context_len+1, (right_context_len + len(sentence))+1
            annotation = (sentence, [])

            for indx, span in enumerate(text['annotations']):
                st_en = span['span']

                if int(st_en['start']) >= left_context_len and int(st_en['end']) <= right_context_len:
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

            for word, tag in zip(words, labels):
                file.write(word + ' ' + tag + '\n')
            file.write('\n')

        if flag == n:
            print('First {} lines executed'.format(n))
            file.close()
            break


if __name__ == "__main__":
    parse_json_line('nerus_lenta.jsonl.gz')




