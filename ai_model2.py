import pandas as pd
import numpy as np
import nflgame
import datetime
from statistical_model import get_team
from calendar import week
from statistical_model import home_model as hm

from sklearn.ensemble import RandomForestClassifier

START_YEAR = 2009
MAX_WEEKS = 17
MIN_WEEK = 200906  # sixth week of 2009


class new_model():
    def __init__(self):
        self.raw_stats = None
    
    
    def load_stats(self):
        stats_map = {}
        # build all of the stuff in the stats map
        remove = set(['pos_time'])
        self.outputs = {'week': [], 'output': []}
        
        stats_map['win'] = []
        stats_map['team'] = []
        stats_map['id'] = []     # to identify a single game
        stats_map['home'] = []
        stats_map['date'] = []
        stats_map['week'] = []
        self.stats =  list(set(nflgame.games(2009, 1)[0].stats_home._fields) - remove)
        print self.stats
        for stat in self.stats:
            stats_map[stat] = []
            
        game_num = 0
        self.last_year, self.last_week = nflgame.live.current_year_and_week() 
        for year in range(START_YEAR, self.last_year+1):
            for week in range(1, self.last_week + 1):
                print ('Year %d, week %d' % (year, week))
                games = nflgame.games(year, week)
                for game in games:
                    play_week = game.schedule['year'] * 100 + game.schedule['week']
                    self.outputs['week'].append(play_week)
                    self.outputs['output'].append(float(get_team(game.winner) == get_team(game.home)))
                    
                    teams = [get_team(game.home), get_team(game.away)]
                    for teamn, team in enumerate(teams):
                        stats_map['win'].append(float(get_team(game.winner) == get_team(team)))
                        stats_map['id'].append(game_num)
                        stats_map['home'].append((teamn + 1) % 2)
                        stats_map['team'].append(get_team(team))
                        stats_map['date'].append(int(game.schedule['eid']) / 100)
                        stats_map['week'].append(play_week)
                    for stat in self.stats:
                        # home
                        stats_map[stat].append(float(game.stats_home.__getattribute__(stat)))
                        stats_map[stat].append(float(game.stats_away.__getattribute__(stat)))
                    
                    game_num +=1
        self.stats.append('win')
        self.stat_data = pd.DataFrame(stats_map)     
        self.outputs = pd.DataFrame(self.outputs)
    
    def process_records(self, alpha= .5):
        #yw = year * 100 + week
        recs = self.stat_data
        teams = set(self.stat_data['team'].unique())   
        for team in teams:
            team_time_series = recs[recs['team'] == team][self.stats]
            team_time_series = team_time_series.ewm(alpha=alpha).mean().shift(1)
            recs.update(team_time_series)
            pass
        
        self.recs = recs
        
        # get home and away
        home_teams = recs[recs['home'] == 1]
        away_teams = recs[recs['away'] == 0]
        
        # combine home and away
        self.game_recs = pd.merge(home_teams[['id','week'] + self.stats], away_teams[['id'] + self.stats], on='id').drop(['id'], axis=1)
        
    
    def train(self, year, week):
        '''
        Train up until year, week
        '''
        yw = year * 100 + week
        valid_recs = self.game_recs[(self.game_recs['week'] <= yw) & (self.game_recs['week'] >= MIN_WEEK)].drop('week', axis=1)
        outputs = self.outputs[(self.outputs['week'] <= yw) & (self.outputs['week'] >= MIN_WEEK)].drop('week', axis=1)
        assert len(valid_recs) == len(outputs)
        
        self.model = RandomForestClassifier(max_depth=2)
        self.model.fit(valid_recs.values, outputs.values)
        pass
    
    def predict(self, home_team, away_team):
        home = self.recs[self.recs['team'] == home_team].iloc[-1][self.stats]
        away = self.recs[self.recs['team'] == away_team].iloc[-1][self.stats]
        
        return self.model.predict(home.append(away).to_frame().values.T)
    
if __name__ ==  '__main__':
    m = new_model()
    m.load_stats()
    m.process_records()
    m.train(2017, 16)
    print m.predict('SEA', 'SF')