import pandas as pd
import numpy as np
import nflgame
import math
from statistical_model import get_team
from calendar import week
from statistical_model import home_model as hm

from sklearn.ensemble import RandomForestClassifier

START_YEAR = 2009
MAX_WEEKS = 17
MIN_YEAR = 2009
MIN_W = 06
MIN_WEEK = MIN_YEAR * 100 + MIN_W  # sixth week of 2009


class new_model():
    def __init__(self):
        self.raw_stats = None
        self.print_it = True
    
    def __get_weird_stats__(self, game, attrib_list, home):
            game_list = [0] * len(attrib_list)
            for player in game.players.filter(home=home):
                for i, attribute in enumerate(attrib_list):
                    if attribute in player.__dict__:
                        game_list[i] += player.__dict__[attribute]
            return game_list
    
    def print_out(self, output):
        if self.print_it:
            print (output)
    
    def load_stats(self):
        stats_map = {}
        # build all of the stuff in the stats map
        remove = set(['pos_time'])
        self.outputs = {'week': [], 'output': []}
        
        reg_attribs = ['win', 'team', 'id', 'home', 'date', 'week', 'score']
        extra_attribs = ['rushing_att', 'passing_att']
        
        for att in extra_attribs:
            stats_map[att] = []
            # defense
            stats_map['d' + att] = []
        
        for att in reg_attribs:
            stats_map[att] = []
        
        self.stats =  list(set(nflgame.games(MIN_YEAR, 1)[0].stats_home._fields) - remove)
        self.print_out( self.stats)
        d_stats = []
        for stat in self.stats:
            stats_map[stat] = []
            stats_map['d' + stat] = []
            d_stats.append('d' + stat)
        
        
        game_num = 0
        self.last_year, self.last_week = nflgame.live.current_year_and_week() 
        for year in range(START_YEAR, self.last_year+1):
            for week in range(1, self.last_week + 1):
                self.print_out ('Year %d, week %d' % (year, week))
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
                    
                    stats_map['score'].append(game.score_home)
                    stats_map['score'].append(game.score_away)
                    
                    for stat in self.stats:
                        # offense
                        stats_map[stat].append(float(game.stats_home.__getattribute__(stat)))
                        stats_map[stat].append(float(game.stats_away.__getattribute__(stat)))
                        
                        # swap order of home away for defense
                        stats_map['d' + stat].append(float(game.stats_away.__getattribute__(stat)))
                        stats_map['d' + stat].append(float(game.stats_home.__getattribute__(stat)))
                        
                    
                    # for home and away get extra stats
                    for h in [True, False]:
                        new_stats = self.__get_weird_stats__(game, extra_attribs, h)
                        for stat, value in zip(extra_attribs, new_stats ):
                            stats_map[stat].append(value)
                            
                            # swap for defense
                            if not h:
                                stats_map['d' + stat].insert(-1, value)
                            else:
                                 stats_map['d' + stat].append(value)
                                 
                    game_num +=1
                    
        self.stats.append('win')
        self.stats.append('score')
        self.stat_data = pd.DataFrame(stats_map)     
        self.outputs = pd.DataFrame(self.outputs)
        
        # transform 
        for d in ['', 'd']:
            self.stat_data[d + 'rush_per_att'] = self.stat_data[d + 'rushing_yds'] / self.stat_data[d + 'rushing_att']
            self.stat_data[d + 'pass_per_att'] = self.stat_data[d + 'passing_yds'] / self.stat_data[d + 'passing_att']
        
        # drop colls
        self.stats += (['rush_per_att', 'pass_per_att'] + ['drush_per_att', 'dpass_per_att'] + d_stats)
        drop_list = ['rushing_att', 'passing_yds', 'passing_att']
        self.stats = list(set(self.stats) - set(drop_list))
        
        self.old_stats = self.stats
    
    def process_records(self, alpha= .5):
        #yw = year * 100 + week
        self.stats = self.old_stats[:]
        self.print_out('Processing records..')
        
        recs = self.stat_data.copy(True)
        teams = set(self.stat_data['team'].unique())   
        for team in teams:
            team_time_series = recs[recs['team'] == team][self.stats]
            team_time_series = team_time_series.ewm(alpha=alpha).mean().shift(1)
            recs.update(team_time_series)
            pass
        
        self.recs = recs

        self.print_out('Correlations:')
        win_correlation = self.recs.corr()['win']
        self.print_out (win_correlation)
        
        for col, val in win_correlation.iteritems():
            if abs(val) < .2 and col in self.stats:
                self.print_out('Removing %s' % col)
                self.stats.remove(col)
        
        self.print_out('Using columns: %s' % str(self.stats))
        self.print_out(self.recs[['win', 'team']][-32:])
        # get home and away
        home_teams = recs[recs['home'] == 1]
        away_teams = recs[recs['home'] == 0]
        
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
        
        self.model = RandomForestClassifier(max_depth=3)
        self.model.fit(valid_recs.values, outputs.values)
        pass
    
    def predict(self, home_team, away_team):
        home = self.recs[self.recs['team'] == home_team].iloc[-1][self.stats]
        away = self.recs[self.recs['team'] == away_team].iloc[-1][self.stats]
        
        return self.model.predict(home.append(away).to_frame().values.T)
    
    def predict_week(self, year, week):
        yw = year * 100 + week
        game_set = self.game_recs[self.game_recs['week'] == yw].drop('week', axis=1)
        return self.model.predict(game_set.values)
    
    def actual_week(self, year, week):
        yw = year * 100 + week
        game_set = self.outputs[self.outputs['week'] == yw].drop('week', axis=1)
        return game_set.values
    
    def back_test(self):
        psum = 0
        i = 0
        self.print_out('Running backtest...')
        
        for year in range(MIN_YEAR + 2, self.last_year + 1):
            year_acc = 0
            for week in range(1, MAX_WEEKS + 1):
                i += 1
                if year == MIN_YEAR and week < MIN_W or week == MAX_WEEKS and year == self.last_year:
                    continue
                self.train(year, week)
                next_year, next_week = (year, week + 1) if week < MAX_WEEKS else (year + 1, week)
                predict = self.predict_week(next_year, next_week)
                actual = self.actual_week(next_year, next_week)
                acuracy =1.0 * np.sum(predict == actual.T) / len(predict)
                psum += acuracy
                year_acc += acuracy
                #self.print_out('Year %d, week %d: %f' % (year, week, acuracy))
            self.print_out('Accuracy Year %d: %f' % ( year ,year_acc / MAX_WEEKS))
        ave_acc = (psum / i)
        self.print_out('Prediction accuracy is %f' % ave_acc)
        return ave_acc
    
    def find_best_alpha(self):
        max_acc =0 
        max_alpha = 0
        for alpha in np.linspace(.1, .9, 100):
            self.print_it = False
            self.process_records(alpha)
            acc = self.back_test()
            self.print_it = True
            self.print_out('Alpha %f, accuracy: %f' % (alpha, acc))
            if acc > max_acc:
                max_acc = acc
                max_alpha = alpha
        return max_alpha
    
if __name__ ==  '__main__':
    m = new_model()
    m.load_stats()
    #m.process_records(0.027980)
    m.process_records(0.172)
    #m.train(2017, 17)
    #print m.predict('SEA', 'SF')
    
    m.back_test()
    #malpha = m.find_best_alpha()
    #print 'Best alpha: %f' % malpha
    #print m.back_test()
    '''
    for alpha in range(9):
        print alpha
        m.process_records((alpha + 1.0) / 10, False)
        m.back_test()
    '''