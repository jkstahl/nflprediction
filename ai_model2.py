
import pandas as pd
import numpy as np
import nflgame
import math, pickle, os
from statistical_model import get_team
from calendar import week
import multiprocessing as mp

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble.weight_boosting import AdaBoostClassifier


START_YEAR = 2009
MAX_WEEKS = 17
MIN_YEAR = 2009
MIN_W = 06
MIN_WEEK = MIN_YEAR * 100 + MIN_W  # sixth week of 2009
SAVE_FILENAME = 'savestats.dat'



class new_model():
    stat_data = None
    stats = None
    outputs = None
    
    def __init__(self):
        self.raw_stats = None
        self.print_it = True
    
    def __get_weird_stats__(self, game, attrib_list, home):
            game_list = [0.0] * len(attrib_list)
            for player in game.players.filter(home=home):
                for i, attribute in enumerate(attrib_list):
                    if attribute in player.__dict__:
                        game_list[i] += player.__dict__[attribute]
            return game_list
    
    def print_out(self, output):
        if self.print_it:
            print (output)
    
    def load_stats(self):
        
        
        self.last_year, self.last_week = nflgame.live.current_year_and_week() 
        if os.path.exists(SAVE_FILENAME):
            self.print_out('Loading old stat data..')
            self.stats, self.stat_data, self.outputs, last_year, last_week = pickle.load(open(SAVE_FILENAME))
        
        if not os.path.exists(SAVE_FILENAME) or last_year != self.last_year or last_week != self.last_week:
            self.print_out('Savefile not found.  Loading new stats.')
            stats_map = {}
            # build all of the stuff in the stats map
            remove = set(['pos_time'])
            self.outputs = {'week': [], 'output': []}
            
            reg_attribs = ['win', 'team', 'id', 'home', 'date', 'week', 'score']
            extra_attribs = ['rushing_att', 'passing_att'] #, 'defense_sk'
            
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
            
            for year in range(START_YEAR, self.last_year+1):
                for week in range(1, MAX_WEEKS + 1):
                    if year == self.last_year and week > self.last_week:
                        continue
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
            
            
            self.print_out('Saving data...')
            pickle.dump((self.stats, self.stat_data, self.outputs, self.last_year, self.last_week), open(SAVE_FILENAME, 'w'))
            
        self.old_stats = self.stats
        
    
    def __remove_uncorrelated__(self, df, corr_val):
        usable = []
        win_correlation = df.corr()['win']
        self.print_out (win_correlation)
        
        for col, val in win_correlation.iteritems():
            if abs(val) >= corr_val:
                self.print_out('Removing %s' % col)
                usable.append(col)
        return usable
    
    def __windowed_average__(self, x):
        window = self.pre_compute[len(self.pre_compute)-len(x):] * self.discount_weights[:len(x)]
        return np.sum(window * x) / window.sum()
        

    
    def process_records(self, alpha= .5, season_discount = .75):
        '''
        This filters all of the stats creating records for each game week with all of the average stats.  The stats
        are filtered over all of the data so that older games are discounted on exponential decay.  The season are also 
        discounted so that the previous season are not considered as much as the current season.
        
        alpha - this is the filter constant that represents how much to discount the stats from each previous game.
        season_discount - each season, discount by this coefficient.
        '''
        
        #yw = year * 100 + week
        self.stats = self.old_stats[:]
        self.print_out('Processing records..')
        
        self.alpha = alpha
        
        recs = self.stat_data.copy(True)
        teams = set(self.stat_data['team'].unique())
        
        # compute the filter weights so we don't have to do it for each team.   
        self.pre_compute = np.fliplr([np.exp(np.array(range(len(self.stat_data))) * np.log(1-alpha))])[0]
        self.discount_weights = np.exp(np.array(range(len(self.stat_data))) / 17 * np.log(season_discount))
        
        # keep track of the latest records to make future predictions
        self.latest = {}
        for team in teams:
            team_time_series = recs[recs['team'] == team][self.stats]
            team_time_series = team_time_series.expanding().apply(self.__windowed_average__, raw=True)
            self.latest[team] = team_time_series.iloc[-1] 
            team_time_series = team_time_series.shift(1)
            #t2 = team_time_series.ewm(alpha=alpha).mean().shift(1)
            recs.update(team_time_series)
        
        self.print_out('Correlations:')
        self.stats = list(set(self.__remove_uncorrelated__(recs, .2)) & set(self.stats))

        self.print_out('Using columns: %s' % str(self.stats))
        #self.print_out(recs[['win', 'team']][-32:])
        '''
        # Do pca and reselect
        self.pca = PCA()
        fit = self.pca.fit_transform(recs[self.stats].values)
        fit_df = pd.DataFrame(data=fit, index=range(fit.shape[0]), columns=range(fit.shape[1]))
        fit_df['win'] = recs['win']
        self.stats = self.__remove_uncorrelated__(fit_df, .2)
        #self.stats.remove('win')
        recs = pd.concat([recs, fit_df.drop('win', axis=1)], axis = 1)
        '''
        self.recs = recs
        
        # get home and away
        home_teams = recs[recs['home'] == 1]
        away_teams = recs[recs['home'] == 0]
        
        # combine home and away
        self.game_recs = pd.merge(home_teams[self.stats + ['week', 'id']], away_teams[self.stats + ['id']], on='id').drop(['id'], axis=1)

        
    
    def train(self, year, week):
        '''
        Train up until year, week
        '''
        yw = year * 100 + week
        valid_recs = self.game_recs[(self.game_recs['week'] <= yw) & (self.game_recs['week'] >= MIN_WEEK)].drop('week', axis=1)
        outputs = self.outputs[(self.outputs['week'] <= yw) & (self.outputs['week'] >= MIN_WEEK)].drop('week', axis=1)
        assert len(valid_recs) == len(outputs)
        
        #self.model = RandomForestClassifier(max_depth=5)
        self.model = GaussianNB() 
        self.model.fit(valid_recs.values, outputs.values.T[0])
        
    def get_s_vect(self, team):
        return self.latest[team][self.stats]
    
    def predict(self, home_team, away_team):
        home = self.latest[home_team][self.stats]
        away = self.latest[away_team][self.stats]
        new_frame = home.append(away).to_frame().values.T
        prob = self.model.predict_proba(new_frame) 
        return (prob[0][0], prob[0][1], self.model.predict(new_frame)[0])
    
    def predict_week(self, year, week):
        yw = year * 100 + week
        game_set = self.game_recs[self.game_recs['week'] == yw].drop('week', axis=1)
        return self.model.predict(game_set.values)
    
    def actual_week(self, year, week):
        yw = year * 100 + week
        game_set = self.outputs[self.outputs['week'] == yw].drop('week', axis=1)
        return game_set.values
    
    def back_test(self, reruns = 1):
        psum = 0
        i = 0
        self.print_out('Running backtest...')
        last_full_year = self.last_year if self.last_week == MAX_WEEKS else self.last_year - 1
        
        for year in range(MIN_YEAR + 2, last_full_year + 1):
            year_acc = 0
            for week in range(1, MAX_WEEKS + 1):
                i += 1
                if year == MIN_YEAR and week < MIN_W or week == MAX_WEEKS and year == self.last_year:
                    continue
                rerun_sum = 0
                for run in range(reruns):
                    self.train(year, week)
                    next_year, next_week = (year, week + 1) if week < MAX_WEEKS else (year + 1, 1)
                    predict = self.predict_week(next_year, next_week)
                    actual = self.actual_week(next_year, next_week)
                    acuracy =1.0 * np.sum(predict == actual.T) / len(predict)
                    rerun_sum += acuracy
                acuracy = rerun_sum / reruns
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
        max_s = 0
    
        for alpha in np.linspace(.1, .9, 100):
            for season_discount in np.linspace(.1, .9, 100):
                self.print_it = False
                self.process_records(alpha, season_discount)
                acc = self.back_test()
                self.print_it = True
                self.print_out('Alpha %f, max_s %f,accuracy: %f' % (alpha, season_discount, acc))
                if acc > max_acc:
                    self.print_out('FOUND BEST!')
                    max_s = season_discount
                    max_acc = acc
                    max_alpha = alpha
        return (max_alpha, max_s)
    


def proc_wrapper(o, a,s):
    #print 'Starting wrapper...'
    o.process_records(a, s)
    result = o.back_test()
    #print 'Done.'
    return result


if __name__ ==  '__main__':
    m = new_model()
    m.load_stats()
    #m.process_records(0.027980)
    m.process_records(0.172727, 0.663518)
    #m.train(2017, 17)
    #print m.predict('SEA', 'SF')
    
    m.back_test()
    import sys
    sys.exit(0)
    #print m.find_best_alpha()
    max_acc =0 
    max_alpha = 0
    max_s = 0
    season_discount = np.linspace(.1, .99, 100)
    models = [new_model() for sd in season_discount ]
    for o in models:
        o.load_stats()
        o.print_it=False
    pool = mp.Pool()
    for alpha in np.linspace(.1, .9, 100):
        print 'alpha %f' % alpha
        
        results = [pool.apply_async(proc_wrapper, args=(o, alpha, sd,)) for sd, o in zip(season_discount, models)]
        output = [p.get() for p in results]
        output = zip(output, season_discount)
        output = sorted(output, reverse=True) 
        if output[0][0] > max_acc:
            max_acc = output[0][0]
            max_s = output[0][1]
            max_alpha = alpha
            print 'Found new max: alpha %f, season_discount %f, accuracy %f' % (max_alpha, max_s, max_acc)
        
        print 'Best found: alpha %f, season_discount %f, val %f' % (max_alpha, max_s, max_acc)
    #malpha = m.find_best_alpha()
    #print 'Best alpha: %f' % malpha
    #print m.back_test()
    '''
    for alpha in range(9):
        print alpha
        m.process_records((alpha + 1.0) / 10, False)
        m.back_test()
    '''