import wikipedia as w
from wikipedia.exceptions import WikipediaException
from articles import arr
import pickle as pkl

if __name__ == '__main__':
    summarys = []
    for art in arr:
        try:
            sum = w.summary(str(art))
            print(sum)
            summarys.append(sum)
        except WikipediaException:
            continue
