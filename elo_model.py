#import nflgame
print ('elo_model')
from game_scraper import game_scraper 
import pandas as pd
from statistical_model import get_team
from astropy.units import hd

START_YEAR = 2009
MAX_WEEK = 17
STARTING_ELO = 1500
MAX_ELO = 4000
MIN_ELO = 100


class elo_model():
    
    def __init__(self, last_year_played, last_week_played, nflgame):
        self.nflgame = nflgame
        # These are data frames where the collumns are teams and the rows are year/week
        self.teams = list(set([get_team(team[0]) for team in self.nflgame.teams]))
        self.reset()
        self.last_week_played = last_week_played
        self.last_year_played = last_year_played
        self.k_o = 4
        self.k_d = 4
        self.div = 25
        self.points_offset = 24.2
        self.home_advantage = 25
    
    def set_params(self, params):
        assert (len(params) == 4)
        self.k_o, self.k_d, self.div, self.home_advantage = params
    
    def reset(self):
        self.elo_offense = pd.DataFrame({i : [STARTING_ELO] for i in self.teams}, index=[START_YEAR * 100])
        self.elo_defense = pd.DataFrame({i : [STARTING_ELO] for i in self.teams}, index=[START_YEAR * 100])
    
    def match(self, offense_elo, defense_elo, home):
        '''
        perform a match for the given year and week.  return the expected scores.
        '''
        # get home score which is matching the home offense to the away defense
        return (offense_elo - defense_elo + int(home) * self.home_advantage) / self.div + self.points_offset
    
    def get_elo(self, df, team, yearweek):
        candidates = df[df.index <= yearweek]
        if len (candidates) == 0:
            return STARTING_ELO
        return candidates.iloc[-1][team]
    
    def get_offense_elo(self, team, yearweek):
        # get all years less than given yearweek
        return self.get_elo(self.elo_offense, team, yearweek)

    def get_defense_elo(self, team, yearweek):
        # get all years less than given yearweek
        return self.get_elo(self.elo_defense, team, yearweek)
    
    def __check_yearweek__(self,df, yearweek):
        if yearweek not in df.index:
            df.loc[yearweek] = df.iloc[-1]
    
    def update_team_offense_elo(self, team, yearweek, new_elo):
        self.__check_yearweek__(self.elo_offense, yearweek)
        self.elo_offense[team][yearweek] = new_elo
        
    def update_team_defense_elo(self, team, yearweek, new_elo):
        self.__check_yearweek__(self.elo_defense, yearweek)
        self.elo_defense[team][yearweek] = new_elo
    
    def update(self, yearweek, offense, defense, score, expected):
        oe = self.get_offense_elo(offense, yearweek - 1)
        de = self.get_defense_elo(defense, yearweek - 1)
        self.update_team_offense_elo(offense, yearweek, (oe + self.k_o * (score - expected)).clip(MIN_ELO, MAX_ELO))

        self.update_team_defense_elo(defense, yearweek, (de + self.k_d * (expected - score)).clip(MIN_ELO, MAX_ELO))
    
    def __year_week_combine__(self, year, week):
        return year * 100 + week
    
    def play_match(self, year,week, home, away):
        home = get_team(home)
        away = get_team(away)
        # need to subtract 1 from yearweek because the current yearweek has the results.
        yearweek = self.__year_week_combine__(year, week) -1
        ho = self.get_offense_elo(home, yearweek)
        hd = self.get_defense_elo(home, yearweek)
        ao = self.get_offense_elo(away, yearweek)
        ad = self.get_defense_elo(away, yearweek)
        return (self.match(ho, ad, True), self.match(ao, hd, False))
        
    def load_all_matches(self, verbose=False):
        self.reset()
        for year in range(START_YEAR, self.last_year_played + 1):
            if verbose:
                print ('Parsing year %d' % year)
            for week in range(1, self.last_week_played + 1):
                if verbose:
                    print ('week %d' % week)
                yearweek = self.__year_week_combine__(year, week)
                for game in self.nflgame.games(year, week):
                    if verbose:
                        print (game)
                    home, away = get_team(game.home), get_team(game.away)

                    # update the elo based on the score of the home and away 
                    for t1, t2, score, homet in [(home, away, game.score_home, True), (away, home, game.score_away, False)]:
                        if score == None:
                            break
                        expected = self.match(self.get_offense_elo(t1, yearweek-1), self.get_defense_elo(t2, yearweek-1), homet)
                        
                        self.update(yearweek, t1, t2, score, expected)
                        if verbose:
                            print ('home: %s away: %s score: %d expected %d' % (home, away, score, expected) )

                    
