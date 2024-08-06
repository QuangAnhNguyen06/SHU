import pandas
import numpy
import librosa
from matplotlib import pyplot as plt
import librosa.display
import allosaurus.app
import panphon
import itertools
import seaborn
import matplotlib.transforms



def parse_timestamp_output(output):
    records = []
    lines = output.split('\n')
    for line in lines:
        tok = line.split(' ')
        start = float(tok[0])
        duration = float(tok[1])
        end = start + duration
        label = tok[2]
        records.append(dict(start=start, end=end, label=label))
    df = pandas.DataFrame.from_records(records)
    return df

def is_consonant(ipa):
    ft = panphon.FeatureTable()
    cons = ft.word_fts(ipa)[0] >= {'cons': 1}
    return cons

def phone_recognize_file(path, emit=1.2, lang='eng'):
    model = allosaurus.app.read_recognizer()
    out = model.recognize(path, lang, timestamp=True, emit=emit)
    phones = parse_timestamp_output(out)
    phones['consonant'] = phones['label'].apply(is_consonant)
    phones['color'] = phones['consonant'].replace({True: 'green', False: 'red'})
    return phones

def load_spectrogram(path, hop_length=1024, sr=16000, n_mels=64, ref=numpy.max, **kwargs):
    y, sr = librosa.load(path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, **kwargs)
    S_db = librosa.power_to_db(S, ref=ref)
    tt = numpy.arange(0, S_db.shape[1]) * (hop_length / sr)
    print(tt.shape)
    df = pandas.DataFrame(S_db.T, index=pandas.Series(tt, name='time'))
    return df

def plot_events(ax, df, start='start', end='end', color=None, annotate=None,
                label=None, alpha=0.2, zorder=-1,
                text_kwargs={}, **kwargs):


    palette = itertools.cycle(seaborn.color_palette())

    def valid_time(dt):
        return not pandas.isnull(dt)

    for idx, row in df.iterrows():
        s = row[start]
        e = row[end]

        if color is None:
            c = next(palette)
        else:
            c = row[color]

        label = None
        if valid_time(s) and valid_time(e):
            ax.axvspan(s, e, label=label, color=c, alpha=alpha, zorder=zorder)
        if valid_time(e):
            ax.axvline(e, label=label, color=c, alpha=alpha, zorder=zorder)
        if valid_time(s):
            ax.axvline(s, label=label, color=c, alpha=alpha, zorder=zorder)

        
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        if annotate is not None:
            ax.text(s, 1.05, row[annotate], transform=trans, **text_kwargs)



