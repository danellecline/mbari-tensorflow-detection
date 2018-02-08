import utils
import os
import conf

if __name__ == '__main__':

    collections = ['MBARI_BENTHIC_2017_small']

    for c in collections:
        dir = os.path.join(os.getcwd(), 'data', c)
        train_per = 0.95
        tests_per = 0.05
        utils.split(dir, train_per, tests_per)

    print('Done')




