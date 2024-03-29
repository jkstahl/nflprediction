
import pandas as pd
import numpy as np

import math, pickle, os
from statistical_model import get_team
from calendar import week
import multiprocessing as mp
from game_scraper2 import game_scraper
from sklearn.ensemble import RandomForestClassifier
#from ai_model2 import output


START_YEAR = 2009
MAX_WEEKS = 17
MIN_YEAR = 2009
MIN_W = 6
MIN_WEEK = MIN_YEAR * 100 + MIN_W  # sixth week of 2009
SAVE_FILENAME = 'savestats.dat'



class deap_learner_model():
    stat_data = None
    stats = None
    outputs = None
    
    def __init__(self):
        self.use_columns = ['points_scored', 'result', 'location', 'output']
        self.default_columns =      ['extra_points_attempted',
                            'extra_points_made',
                            'field_goals_attempted',
                            'field_goals_made',
                            'fourth_down_attempts', 
                            'fourth_down_conversions', 
                            'interceptions',
                            'pass_attempts', 
                            'pass_completion_rate', 
                            'pass_completions',
                            'pass_touchdowns', 
                            'pass_yards', 
                            'pass_yards_per_attempt',
                            'points_allowed', 
                            'points_scored',
                            'punt_yards', 
                            'punts',
                            'quarterback_rating', 
                            'rush_attempts', 
                            'rush_touchdowns',
                            'rush_yards', 
                            'rush_yards_per_attempt', 
                            'third_down_attempts',
                            'third_down_conversions', 
                            'times_sacked',
                            'yards_lost_from_sacks',
                            'result',
                            'week',
                            'location',
                            'output']
        self.raw_stats = None
        self.print_it = True
        self.stats = game_scraper(2022).stats()
        pass
        self.transform()
        self.make_vectors()
        #self.fit()
        
    def set_columns(self, cols):
        self.use_columns  = [ self.default_columns[i] for i in cols]
    
    def __windowed_average__(self, x):
        window = self.pre_compute[len(self.pre_compute)-len(x):] 
        return np.sum(window * x) / window.sum()    
    
    def transform(self):
        alpha = .25
        
        self.pre_compute = np.fliplr([np.exp(np.array(range(1000)) * np.log(1-alpha))])[0]
        transformers = [
                            'extra_points_attempted',
                            'extra_points_made',
                            'field_goals_attempted',
                            'field_goals_made',
                            'fourth_down_attempts', 
                            'fourth_down_conversions', 
                            'interceptions',
                            'pass_attempts', 
                            'pass_completion_rate', 
                            'pass_completions',
                            'pass_touchdowns', 
                            'pass_yards', 
                            'pass_yards_per_attempt',
                            'points_allowed', 
                            'points_scored',
                            'punt_yards', 
                            'punts',
                            'quarterback_rating', 
                            'rush_attempts', 
                            'rush_touchdowns',
                            'rush_yards', 
                            'rush_yards_per_attempt', 
                            'third_down_attempts',
                            'third_down_conversions', 
                            'times_sacked',
                            'yards_lost_from_sacks',
                            'result',
                            ]
        others = {    
                            'result' : lambda x : 1 if x == 'Win' else 0,
                            'location'  : lambda x : 1 if x == 'Home' else 0,
                            'week' : lambda x : x / 17,
                    } 
        for team, df in self.stats.items():
            for o, f in others.items():
                df[o] = df[o].apply(f)
            df['output'] = df['result']
        for team, df in self.stats.items():
            for col in transformers:
                df[col] = df[col].expanding().apply(self.__windowed_average__, raw=True)
        for team in self.stats:
            self.stats[team] = self.stats[team].iloc[7:]
            outputs = self.stats[team]['output']
            self.stats[team] = self.stats[team].iloc[:-1]
            
            self.stats[team]['output'] = outputs[1:].set_axis(self.stats[team].index)
            pass
    
    def make_vectors(self):
        # stack all of the teams
        self.vects = pd.DataFrame()
        for team, df in self.stats.items():
            self.vects = self.vects.append(df,)
        
        self.vects = self.vects[self.use_columns]
        
        self.vects = self.vects.loc[self.vects['location'] == 1].join(self.vects.loc[self.vects['location'] == 0], rsuffix='_away')
        # pull out home and away games and put them together
        #self.outputs = self.vects['output']
        #outputs = self.vects['output']
        self.vects.drop(['output_away', 'location', 'location_away'], axis=1, inplace=True)

        pass
    
    def fit(self):
        # shuffle rows pull out train set and val set
        train, validation = np.split(self.vects.sample(frac=1), [int(.8*len(self.vects))])

        train = train.dropna()
        validation = validation.dropna()

        # train
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(train.drop('output', axis=1), train['output'])
        
        # validate on val set
        return sum(clf.predict(validation.drop('output', axis=1)) == validation['output']) / len (validation)
        

if __name__ == '__main__':  
    
    dl = deap_learner_model()
    #print (dl.stats['SEA'].columns)
    dl.set_columns(range(len(dl.default_columns)))
    print(dl.fit())
    