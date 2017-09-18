import pandas as pd
import numpy as np
import nflgame
from modeler2 import get_team
from calendar import week

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import GaussianNB
import pickle, os
from pandas.core.frame import DataFrame


FIRST_YEAR = 2009
LAST_YEAR = 2017
LAST_WEEK = 1
SAVE_FILE = 'formatted_data'

DEBUG=True

class stat_tracker():
    def __init__(self, columns):
        # One set per team for this stat.  If a new vector is added, append to the old stat
        self.columns = columns
        self.stat = pd.DataFrame(columns=columns + ['team', 'home', 'opponent', 'date'])
        self.teams = set([])
    
    '''
    def get_teams(self):
        return pd.unique(self.stat['team'])
    '''
        
    def add_stat(self, team, stat,home, date, opponent=None ):
        '''
        append stat to end of the list for team
        '''
        self.teams.add(team)
        self.stat.loc[len(self.stat)] = stat + [team, home, opponent, date]
    
    def get_stats(self, team):
        return self.stat[self.stat['team'] == team][self.columns].values
    
    def get_stats_mean(self, team, date, go_back):
        team_test = self.stat['team'] == team
        date_test = self.stat['date'] <= date
        records = self.stat[team_test & date_test][self.columns][-1*go_back:]
        weights = np.array([[1.0] * len (self.columns)] * records.shape[0])
        week = date % 100
        if week < go_back:
            weights[0:go_back-week] = weights[0:go_back-week] * .5
        #print ('Date: %d, weights: %s' % (date, str(weights)))
        return (records * weights).sum() / weights.sum(axis=0)
    
    def get_means(self, date, go_back):
        '''
        Return a dataframe of means per team.
        '''
        teams = np.unique(self.stat['team'])
        means = pd.DataFrame()
        for team in teams:
            means = pd.concat([means, self.get_stats_mean(team, date, go_back)], axis=1)
        means = means.T
        means['team'] = teams
        return means
        
    
    def get_opponents(self, team):
        return self.stat[self.stat['team'] == team]['opponent'].values
    
    def get_opponent_stats(self, team):
        return self.stat[self.stat['opponent'] == team][self.columns].values
    
    def get_teams(self):
        return list(self.teams)
    
    def get_defense_means(self, team, date, go_back, means):
        '''
        Find the mean difference that this team makes the other team from average.
        '''
        team_test = self.stat['team'] == team
        date_test = self.stat['date'] <= date
        real_stats = self.stat[team_test & date_test][-1*go_back:].reset_index()     
        merged = pd.merge(real_stats[['opponent']], means, how='left', left_on='opponent', right_on='team')
        return (real_stats[self.columns] - merged[self.columns]).mean()


    
class ai_model():
    data_storage = None
    save_games = None
    
    def __get_final_stats__(self):
        return ['rush_att_per_game', 'pass_att_per_game', 'yards_per_rush', 'yards_per_pass', 'points_per_yard', 'points_per_game', 'turnovers', 'win',
                'rush_att_per_game_def', 'pass_att_per_game_def', 'yards_per_rush_def', 'yards_per_pass_def', 'points_per_yard_def', 'points_per_game_def', 'turnovers_def']
        #return ['yards_per_rush', 'yards_per_pass', 'points_per_yard', 'points_per_game', 'win',
        #        'yards_per_rush_def', 'yards_per_pass_def', 'points_per_yard_def', 'points_per_game_def']
        
    def __get_player_raw__(self):
        return ['rushing_att', 'passing_att', 'rushing_yds', 'receiving_yds']
    
    def __get_team_raw__(self):
        return ['points']
    
    def __transform_stats__(self, game_list):
        game_list[4] = 1.0 * game_list[4] / (game_list[2] + game_list[3])
        game_list[3] = 1.0 * game_list[3] / game_list[1]
        game_list[2] = 1.0 * game_list[2] / game_list[0]
        #game_list = game_list[2:]
        return game_list
    
    def add_game_static(self, game, year, week):
        accum_stats = self.__get_player_raw__()
        date = year * 100 + week
        # keep track of attemps, yards, pts
        # [home[attempts, yards, pts], away[attempts, yards, pts]]
        teams = [get_team(game.away), get_team(game.home)]
        scores = [ game.score_away ,game.score_home]
        global_stats = [ game.stats_away, game.stats_home]
        stats = [0, 0]

        for homet in [True, False]:
            
            homen = int(homet)
            # keep track of rush attemps, pass attempts, rush yards, pass yards, pts
            game_list = [0]*len(accum_stats)
            for player in game.players.filter(home=homet):
                for i, attribute in enumerate(accum_stats):
                    if attribute in player.__dict__:
                        game_list[i] += player.__dict__[attribute]
            
            game_list += [scores[homen], scores[homen], int(game.winner == teams[homen])]
            
            
            # compute yards per att and pts per yard
            game_list = self.__transform_stats__(game_list)
            game_list.append(global_stats[homen].turnovers)
            stats[homen] = game_list
            
            
        for homet in [True, False]:
            homen = int(homet)        
            # add game stats to global stats
            otheri =  (homen + 1) % 2
            other_team = get_team(teams[otheri])        
            
            team = get_team(teams[homen])
            
            ai_model.data_storage.add_stat(team, stats[homen] + stats[otheri][:-1], homet, date, other_team )  # pull off wins      
    
    def __get_vector__(self, home, away, date, go_back):
        home_stats = ai_model.data_storage.get_stats_mean(home, date, go_back)
        away_stats = ai_model.data_storage.get_stats_mean(away, date, go_back)
        
        #home_def_stats
        #home_defense = ai_model.data_storage.get_defense_means(home, date, go_back, means)
        #away_defense = ai_model.data_storage.get_defense_means(away, date, go_back, means)
        
        return pd.concat([home_stats, away_stats], join='inner').to_frame().T

                

    def __build_records__(self, go_back):
        if go_back != self.go_back:  # don't rebuild if we already have this
            # build game records
            self.Y = pd.DataFrame([])
            self.X = pd.DataFrame([])
            dates = []
            
            for i in range(go_back, len(ai_model.save_games)):
                year, week, games = ai_model.save_games[i]
                if DEBUG:
                    print ('Creating records: %d, %d' % (year, week))
                date = year * 100 + week
                #means = ai_model.data_storage.get_means(date, go_back)
                for home, away, score_home, score_away in games:
                    dates.append(date)
                    home_win = int(score_home > score_away)   # 1 for home win 0 otherwise
                    self.Y = self.Y.append(pd.DataFrame([home_win]), ignore_index=True)
                    
                    # make sure date - 1 so we don't get same-day stats
                    new_vect = self.__get_vector__(home, away, date - 1, go_back) 
                    self.X = pd.concat([self.X, new_vect], ignore_index=True)
                    pass

                
            dates = np.array(dates)
            self.Y['date'] = pd.Series(dates, index = self.Y.index)
            self.X['date'] = pd.Series(dates, index = self.X.index)

    
    def pull_data(self):
        
        # pull all stats from 2009 week 1 to now 2017 week 1
        if ai_model.data_storage == None:
            year, week = 0, 0
            if os.path.exists(SAVE_FILE):
                print ('Loading previous data...')
                with open(SAVE_FILE) as ri:
                    year, week, ai_model.data_storage, ai_model.save_games = pickle.load(ri)
            
            if year != LAST_YEAR or week != LAST_WEEK:
                print ('Loading new game data...')
                ai_model.data_storage = stat_tracker(self.__get_final_stats__())
                ai_model.save_games = []
                last_week = 17
                for year in range(FIRST_YEAR, LAST_YEAR + 1):
                    if year == LAST_YEAR: last_week = LAST_WEEK
                    for week in range(1, last_week + 1):
                        print ('year %d, week %d' % (year, week))
                        games = nflgame.games(year, week, kind = 'REG')
                        new_games = []
                        for game in games:
                            self.add_game_static(game, year, week)
                            new_games.append((get_team(game.home), get_team(game.away), game.score_home, game.score_away))
                        ai_model.save_games.append((year, week, new_games))
                print ('Pickling...')
                with open(SAVE_FILE, 'w') as wo:
                    pickle.dump((LAST_YEAR, LAST_WEEK, ai_model.data_storage, ai_model.save_games), wo)
                print ('Saved data.')
           
            
                    
                    
    def __init__(self):
        self.X = None
        self.Y = None
        
        self.cur_date = None
        self.go_back = None
        
        self.game_records = pd.DataFrame(columns=[])
        self.pull_data()
        

        pass
    
    def process_history(self, start_year, end_year, start_week, end_week):
        go_back = (end_year - start_year + 1) * 17 - (start_week - 1) - (17 - end_week)
        self.__build_records__(go_back)
        self.go_back = go_back
        # pull inputs and outputs from end year and end week and train the model
        end_date = end_year * 100 + end_week
        X = self.X[self.X['date'] <= end_date].ix[:, self.X.columns != 'date'].values
        Y = self.Y[self.Y['date'] <= end_date].ix[:, self.Y.columns != 'date'].values.T[0]
        
        self.cur_date = end_date
        
        # train model
        #self.model = RandomForestClassifier(max_depth=2, random_state=0)
        #self.model = AdaBoostClassifier()
        #self.model = GaussianNB()
        #self.model = KNeighborsClassifier(3)
        self.model = MLPClassifier(alpha=1)
        
        self.model.fit(X, Y)
        #print str(zip(ai_model.data_storage.columns + ai_model.data_storage.columns, self.model.feature_importances_))
        pass
        
    def play_match(self, home, away, time, weather, sims):
        vect = self.__get_vector__(home, away, self.cur_date, self.go_back)
        probs = self.model.predict_proba(vect)
        
        return (0, 0, probs[0][1], probs[0][0])
        
        