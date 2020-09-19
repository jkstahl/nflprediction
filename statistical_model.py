import nflgame
import numpy as np
import pandas as pd

game_cache = {}

team_map = {
            'STL' : 'LA',
            'JAC' : 'JAX',
            'SD'  : 'LAC',
            'STL' : 'PIT',
            }


class stat_tracker():
    def __init__(self, columns):
        # One set per team for this stat.  If a new vector is added, append to the old stat
        self.columns = columns
        self.stat = pd.DataFrame(columns=columns + ['team', 'home', 'opponent'])
        self.teams = set([])
        
    
    def add_stat(self, team, stat,home, opponent=None ):
        '''
        append stat to end of the list for team
        '''
        self.teams.add(team)
        self.stat.loc[len(self.stat)] = stat + [team, home, opponent]
    
    def get_stats(self, team):
        return self.stat[self.stat['team'] == team][self.columns].values
    
    def get_opponents(self, team):
        return self.stat[self.stat['team'] == team]['opponent'].values
    
    def get_opponent_stats(self, team):
        return self.stat[self.stat['opponent'] == team][self.columns].values
    
    def get_home_means(self, team):
        teams = self.stat['team'] == team
        homes = self.stat['home'] ==True
        return self.stat[teams & homes][self.columns].values
    
    def get_away_means(self, team):
        teams = self.stat['team'] == team
        homes = self.stat['home'] == False
        return self.stat[teams & homes][self.columns].values
    
    def get_teams(self):
        return list(self.teams)

def get_team(team):
    if team in team_map:
        return team_map[team]
    return team


class team_tracker():
    
    def __init__(self):
        # team : (attempts per game, yds per attempt, pts per yard)
        # (attempts, yds, pts, games)
        self.stats = ['rush_att_per_game', 'pass_att_per_game', 'yards_per_rush', 'yards_per_pass', 'points_per_yard']
        self.global_stat_map = stat_tracker(self.stats)
        #self.global_defense_map = stat_tracker(self.stats)
        #self.global_opponent_list = stat_tracker(self.stats)
        
        
        self.stats_to_index = {j:i for i,j in enumerate(self.stats)}
        self.index_to_stats = {i:j for i,j in enumerate(self.stats)}
        
        self.distro_map_mean = {}
        self.distro_map_std = {}
        
        self.defense_distro_map_mean = {}
        self.defense_distro_map_std = {}
        
    
    def __transform_stats__(self, game_list):
        game_list[4] = 1.0 * game_list[4] / (game_list[2] + game_list[3])
        game_list[3] = 1.0 * game_list[3] / game_list[1]
        game_list[2] = 1.0 * game_list[2] / game_list[0]
        return game_list

    def __get_raw_stats__(self):
        return ['rushing_att', 'passing_att', 'rushing_yds', 'receiving_yds']
    
    def add_game(self, game):
            accum_stats = self.__get_raw_stats__()
            # keep track of attemps, yards, pts
            # [home[attempts, yards, pts], away[attempts, yards, pts]]
            teams = [get_team(game.away), get_team(game.home)]
            scores = [ game.score_away ,game.score_home]
            stats = [0, 0]

            for homet in [True, False]:
                
                homen = int(homet)
                # keep track of rush attemps, pass attempts, rush yards, pass yards, pts
                game_list = [0, 0, 0, 0, scores[homen]]
                for player in game.players.filter(home=homet):
                    for i, attribute in enumerate(accum_stats):
                        if attribute in player.__dict__:
                            game_list[i] += player.__dict__[attribute]
                
                
                # compute yards per att and pts per yard
                game_list = self.__transform_stats__(game_list)
                
                # add game stats to global stats
                otheri =  (homen + 1) % 2
                other_team = get_team(teams[otheri])        
                
                team = get_team(teams[homen])
                self.global_stat_map.add_stat(team, game_list, homet,other_team)
                stats[homen] = game_list
            
            '''        
            # Generate defensive scores
            for homet in [True, False]:
                homen = int(homet)
                otheri =  (homen + 1) % 2
                other_team = teams[otheri]
                other_stats = stats[otheri]
                team = teams[homen]
                
                self.global_defense_map.add_stat(team, other_stats, other_team)
                #self.global_opponent_list.add_stat(team, other_team)
            '''                
                
    def process_history(self, from_year, to_year, from_week, to_week, week_type = 'REG'):
        for year in range(from_year, to_year + 1):
            week_low = from_week if year == from_year else 1
            week_high = to_week if year == to_year else 17
            for week in range(week_low, week_high + 1):
                
                games = nflgame.games(year, week, kind = week_type)
                
                #games = nflgame.games(year, week, kind = week_type)
                for game in games:
                    self.add_game(game)
        
        # generate the distrobution information    
        self.distro_map_mean = {}
        self.distro_map_std  = {}
        for team in self.get_teams():
            stats_martix = np.array(self.global_stat_map.get_stats(team))
            means =stats_martix.mean(0)
            stds = stats_martix.std(0)
            #print ('%s: %s' % (team, str(means)))
            self.distro_map_mean[team] = means
            self.distro_map_std[team] = stds
        
        #print('\nDefense:')
        # calc the defense by how much it changes relative to the mean
        self.defense_distro_map = {}
        for team in self.get_teams():
            real_stats = []
            for oppenent in self.global_stat_map.get_opponents(team):
                real_stats.append(self.distro_map_mean[oppenent])
            real_stats = np.array(real_stats) 
            stats_diff = real_stats - self.global_stat_map.get_opponent_stats(team)
            means = stats_diff.mean(0)
            self.defense_distro_map_mean[team] = means
            self.defense_distro_map_std[team] = stats_diff.std(0)
            #print ('%s: %s' % (team, str(means)))
                    
    def __get_scores__(self, sims, home, away):
        game = {'home' : home, 'away': away}
        
        sim = {}
        
        rai = self.stats_to_index['rush_att_per_game']
        yri = self.stats_to_index['yards_per_rush']
        pai = self.stats_to_index['pass_att_per_game']
        ypi = self.stats_to_index['yards_per_pass']
        pyi = self.stats_to_index['points_per_yard']

        for ha in ['home', 'away']:
            sim[ha] = {}    
            team_name = get_team(game[ha])
            stats_raw_mean = self.distro_map_mean[team_name]
            stats_raw_std = self.distro_map_std[team_name]
            def_stats_raw_mean = self.defense_distro_map_mean[team_name]
            def_stats_raw_std = self.defense_distro_map_std[team_name]
            
            
            # offense
            sim[ha]['off'] = np.random.multivariate_normal(stats_raw_mean, np.diag(np.square(stats_raw_std)), sims).T
    
            # defense
            sim[ha]['def'] = np.random.multivariate_normal(def_stats_raw_mean, np.diag(np.square(def_stats_raw_std)), sims).T
        
        score_map = {}
        home_or_away = ['home', 'away']
        for i in range(len(home_or_away)):
            tt = home_or_away[i]
            ot = home_or_away[(i+1) % 2] 

            #sim[tt]['off'] = sim[tt]['off'].T  # to break off rows easier
            #sim[ot]['def'] = sim[ot]['def'].T
            rush_yards = 1.0 * (sim[tt]['off'][rai] - sim[ot]['def'][rai]) * (sim[tt]['off'][yri] - sim[ot]['def'][yri])
            pass_yards = 1.0 * (sim[tt]['off'][pai] - sim[ot]['def'][pai]) * (sim[tt]['off'][ypi] - sim[ot]['def'][ypi])
            total_yards = rush_yards + pass_yards
            score_map[tt] = total_yards * (sim[tt]['off'][pyi]-sim[ot]['def'][pyi])
        return score_map
    
    def play_match(self, home, away,time, weather, sims):

        score_map = self.__get_scores__(sims, home, away)
                    
        home_ave = sum(score_map['home']) / sims
        away_ave = sum(score_map['away']) / sims
        home_wins=0.0
        away_wins=0.0
        for i in range(sims):
            if score_map['home'][i] > score_map['away'][i]:
                home_wins += 1
            else:
                away_wins += 1
        
        return (home_ave, away_ave, home_wins/sims, away_wins/sims)
        
    def get_offsensive_stats(self, team):
        return self.global_stat_map[team]
    
    def get_average(self, team, stat):
        return self.distro_map[team][0][self.stats_to_index[stat]]
    
    def get_teams(self):
        return self.global_stat_map.get_teams()
    
    def get_teams_faced(self, team):
        return self.global_opponent_list.get_stats(team)
    
    def get_defense_stats(self, team):
        return self.global_defense_map.get_stats(team)
    
class home_model(team_tracker):
    def __init__(self):
        team_tracker.__init__(self)
        # keep track of all home and away games in the list for each team
        self.home_means = {}
        self.home_stds = {}
        
        self.away_means = {}
        self.away_stds = {}
    

    def process_history(self, from_year, to_year, from_week, to_week, week_type = 'REG'):
        team_tracker.process_history(self, from_year, to_year, from_week, to_week, week_type)
        for team in self.global_stat_map.get_teams():
            means = self.distro_map_mean[team]
            home_games = self.global_stat_map.get_home_means(team)
            self.home_means[team] = (home_games - means).mean(0)
            self.home_stds[team] = (home_games - means).std(0)
            
            away_games = self.global_stat_map.get_away_means(team)
            self.away_means[team] = (away_games - means).mean(0)
            self.away_stds[team] = (away_games - means).std(0)        

    def __get_scores__(self, sims, home, away):
        #team_tracker.__get_scores__(self, sims, home, away)
        
        game = {'home' : home, 'away': away}
        
        sim = {}
        
        rai = self.stats_to_index['rush_att_per_game']
        yri = self.stats_to_index['yards_per_rush']
        pai = self.stats_to_index['pass_att_per_game']
        ypi = self.stats_to_index['yards_per_pass']
        pyi = self.stats_to_index['points_per_yard']

        for ha in ['home', 'away']:
            sim[ha] = {}    
            team_name = get_team(game[ha])
            stats_raw_mean = self.distro_map_mean[team_name]
            stats_raw_std = self.distro_map_std[team_name]
            def_stats_raw_mean = self.defense_distro_map_mean[team_name]
            def_stats_raw_std = self.defense_distro_map_std[team_name]
            
            
            # offense
            sim[ha]['off'] = np.random.multivariate_normal(stats_raw_mean, np.diag(np.square(stats_raw_std)), sims).T
    
            # defense
            sim[ha]['def'] = np.random.multivariate_normal(def_stats_raw_mean, np.diag(np.square(def_stats_raw_std)), sims).T
            
            # home
            sim[ha]['home'] = np.random.multivariate_normal(self.home_means[team_name], np.diag(np.square(self.home_stds[team_name])), sims).T
            
            # away
            sim[ha]['away'] = np.random.multivariate_normal(self.away_means[team_name], np.diag(np.square(self.away_stds[team_name])), sims).T
        
        score_map = {}
        home_or_away = ['home', 'away']
        for i in range(len(home_or_away)):
            tt = home_or_away[i]
            ot = home_or_away[(i+1) % 2] 

            #sim[tt]['off'] = sim[tt]['off'].T  # to break off rows easier
            #sim[ot]['def'] = sim[ot]['def'].T
            rush_yards = 1.0 * (sim[tt]['off'][rai] - sim[ot]['def'][rai] + sim[tt][tt][rai]) * (sim[tt]['off'][yri] - sim[ot]['def'][yri] + sim[tt][tt][yri])
            pass_yards = 1.0 * (sim[tt]['off'][pai] - sim[ot]['def'][pai] + sim[tt][tt][pai]) * (sim[tt]['off'][ypi] - sim[ot]['def'][ypi] + sim[tt][tt][ypi])
            total_yards = rush_yards + pass_yards
            score_map[tt] = total_yards * (sim[tt]['off'][pyi] - sim[ot]['def'][pyi] + sim[tt][tt][pyi])
            
        return score_map