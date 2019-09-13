import nflgame
import numpy as np

team_map = {
            'STL' : 'LA',
            'JAC' : 'JAX',
            'SD'  : 'LAC'
            }


class stat_tracker():
    def __init__(self):
        # One set per team for this stat.  If a new vector is added, append to the old stat
        self.stat = {}
    
    def add_stat(self, team, stat):
        '''
        append stat to end of the list for team
        '''
        if team in self.stat:
            self.stat[team].append(stat)
        else:
            self.stat[team] = [stat]
    
    def get_stats(self, team):
        return self.stat[team]
    
    def get_teams(self):
        return self.stat.keys()

def get_team(team):
    if team in team_map:
        return team_map[team]
    return team


class team_tracker():
    
    def __init__(self):
        # team : (attempts per game, yds per attempt, pts per yard)
        # (attempts, yds, pts, games)
        self.global_stat_map = stat_tracker()
        self.global_defense_map = stat_tracker()
        self.global_opponent_list = stat_tracker()
        
        self.stats = ['rush_att_per_game', 'pass_att_per_game', 'yards_per_rush', 'yards_per_pass', 'points_per_yard']
        self.stats_to_index = {j:i for i,j in enumerate(self.stats)}
        self.index_to_stats = {i:j for i,j in enumerate(self.stats)}
        
        self.distro_map = {}
        self.defense_distro_map = {}
    
    def __transform_stats__(self, game_list):
        game_list[4] = 1.0 * game_list[4] / (game_list[2] + game_list[3])
        game_list[3] = 1.0 * game_list[3] / game_list[1]
        game_list[2] = 1.0 * game_list[2] / game_list[0]
        return game_list
    
    def add_game(self, game):
            home_map = {game.home: 0, game.away: 1}
            accum_stats = ['rushing_att', 'passing_att', 'rushing_yds', 'receiving_yds']
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
                team = get_team(teams[homen])
                self.global_stat_map.add_stat(team, game_list)
                
                stats[homen] = game_list
        
            # Generate defensive scores
            for homet in [True, False]:
                homen = int(homet)
                otheri =  (homen + 1) % 2
                other_team = teams[otheri]
                other_stats = stats[otheri]
                team = teams[homen]
                
                self.global_defense_map.add_stat(team, other_stats)
                self.global_opponent_list.add_stat(team, other_team)
                
                
    def process_history(self, from_year, to_year, from_week, to_week, week_type = 'REG'):
        for year in range(from_year, to_year + 1):
            week_low = from_week if year == from_year else 1
            week_high = to_week if year == to_year else 17
            for week in range(week_low, week_high + 1):
                games = nflgame.games(year, week, kind = week_type)
                for game in games:
                    self.add_game(game)
        
        # generate the distrobution information    
        self.distro_map = {}    
        for team in self.get_teams():
            stats_martix = np.array(self.global_stat_map.get_stats(team))
            means =stats_martix.mean(0)
            stds = stats_martix.std(0)
            #print ('%s: %s' % (team, str(means)))
            self.distro_map[team] = (means, stds)
        
        #print('\nDefense:')
        # calc the defense by how much it changes relative to the mean
        self.defense_distro_map = {}
        for team in self.get_teams():
            real_stats = []
            for oppenent in self.global_opponent_list.get_stats(team):
                real_stats.append(self.distro_map[oppenent][0])
            real_stats = np.array(real_stats) 
            stats_diff = real_stats - self.global_defense_map.get_stats(team)
            means = stats_diff.mean(0)
            self.defense_distro_map[team] = (means, stats_diff.std(0))
            #print ('%s: %s' % (team, str(means)))
                    
    def __get_scores__(self, sim, sims, home, away):
        score_map = {}
        home_or_away = ['home', 'away']
        for i in range(len(home_or_away)):
            tt = home_or_away[i]
            ot = home_or_away[(i+1) % 2] 
            rush_yards = 1.0 * (sim[tt]['off']['rush_att_per_game'] - sim[ot]['def']['rush_att_per_game']) * (sim[tt]['off']['yards_per_rush'] - sim[ot]['def']['yards_per_rush'])
            pass_yards = 1.0 * (sim[tt]['off']['pass_att_per_game'] - sim[ot]['def']['pass_att_per_game']) * (sim[tt]['off']['yards_per_pass'] - sim[ot]['def']['yards_per_pass'])
            total_yards = rush_yards + pass_yards
            score_map[tt] = total_yards * (sim[tt]['off']['points_per_yard']-sim[ot]['def']['points_per_yard'])
        return score_map
    
    def play_match(self, home, away,time, weather, sims):
        game = {'home' : home, 'away': away}
        sim = {}
        for ha in ['home', 'away']:
            sim[ha] = {}    
            team_name = get_team(game[ha])
            stats_raw = self.distro_map[team_name]
            def_stats_raw = self.defense_distro_map[team_name]
            
            
            # offense
            sim[ha]['off'] = {}
            for stat in self.stats:
                stat_index = self.stats_to_index[stat]
                sim[ha]['off'][stat] = np.random.normal(stats_raw[0][stat_index], stats_raw[1][stat_index], sims)
    
            # defense
            sim[ha]['def'] = {}
            for stat in self.stats:
                stat_index = self.stats_to_index[stat]
                sim[ha]['def'][stat] = np.random.normal(def_stats_raw[0][stat_index], def_stats_raw[1][stat_index], sims)

        
        score_map = self.__get_scores__(sim, sims, home, away)
            
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
        self.home_points = {}
        self.away_points = {}
    
    def add_game(self, game):
        team_tracker.add_game(self, game)
        home = get_team(game.home)
        away = get_team(game.away)
        if game.home in self.home_points:
            self.home_points[home].append(game.home_score)
        else:
            self.home_points[game.home] = [game.home_score]
            
        if game.away in self.away_points:
            self.away_points[away].append(game.away_score)
        else:
            self.away_points[away] = [game.away_score]

    def process_history(self, from_year, to_year, from_week, to_week, week_type='REG'):
        team_tracker.process_history(self, from_year, to_year, from_week, to_week, week_type=week_type)
        
        # find mean differece for home and away
        self.home_delta = {}
        self.away_delta = {}
        for team in self.get_teams():
            if team in self.home_points:
                home_scores = np.array(self.home_points[team])
            else:
                home_scores = np.array([])
            if team in self.away_points:
                away_scores = np.array(self.away_points[team])
            else:
                away_scores = np.array([])
            mean = np.append(home_scores, away_scores).mean()  
            self.home_delta[team] = ((home_scores - mean).mean(),(home_scores - mean).std()) 
            self.away_delta[team] = ((away_scores - mean).mean(), (away_scores - mean).std())
            
    def __get_scores__(self, sim, sims, home, away):
        score_map = team_tracker.__get_scores__(self, sim, sims, home, away)
        home_delta = np.random.normal(self.home_delta[home][0], self.home_delta[home][1])
        away_delta = np.random.normal(self.away_delta[away][0], self.away_delta[away][0])
        score_map['home'] += home_delta
        score_map['away'] += away_delta
        return score_map