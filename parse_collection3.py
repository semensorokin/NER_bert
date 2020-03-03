from os import listdir
from os.path import isfile, join
from nltk.tokenize import wordpunct_tokenize

def parse_ann(f_path, path):
    lines = [(i.split('\t')[1].split(' ')[0], i.split('\t')[2][:-1]) for i in open(path + f_path).readlines()]
    annotation = []
    for tag, word in lines:
        annotation.append({'type' : tag, 'text' : word})
    return annotation

def parse_collection3(path = './Collection3/', save_to = 'tarin_colllection3.txt', max_seq_len = 50):
    texts_files = [f for f in listdir(path) if isfile(join(path, f)) and f.endswith('.txt')]

    file = open(save_to, 'w')
    for text in texts_files:
        content = open(path + text).read().replace('\n\n', ' ')
        anns = parse_ann((text[:-4] + '.ann'), path)

        start = 0
        labels = []
        words = []

        for ann in anns:
            span_start = content.find(ann['text'])
            piece = content[start:span_start]
            ws = wordpunct_tokenize(piece)
            words += ws + [ann['text']]
            labels += (['O'] * len(ws)) + [ann['type']]
            start = span_start + len(ann['text'])


        piece = text[start:]
        ws = wordpunct_tokenize(piece)
        words += ws
        labels += (['O'] * len(ws))

        assert len(labels) == len(words), 'Mismatch len labels ans len sentence'

        for i in range(0, len(words), max_seq_len):
            part_wods = words[i:i+max_seq_len]
            part_labels = labels[i:i+max_seq_len]

            for word, tag in zip(part_wods, part_labels):
                file.write(word + ' ' + tag + '\n')
            file.write('\n')
    file.close()

if __name__=="__main__":
    parse_collection3()




