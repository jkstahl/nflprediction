import nflgame
import pandas as pd
from statistical_model import get_team

START_YEAR = 2009
MAX_WEEK = 17
STARTING_ELO = 1500


class elo_model():
    
    def __init__(self, last_year_played, last_week_played):
        # These are data frames where the collumns are teams and the rows are year/week
        self.teams = [get_team(team[0]) for team in nflgame.teams]
        self.elo_offense = pd.DataFrame({i : [STARTING_ELO] for i in self.teams}, index=[START_YEAR * 100])
        self.elo_defense = pd.DataFrame({i : [STARTING_ELO] for i in self.teams}, index=[START_YEAR * 100])
        self.last_week_played = last_week_played
        self.last_year_played = last_year_played
        self.k_o = 4
        self.k_d = 4
        self.div = 25
        self.points_offset = 24.2
        print (self.teams)
        pass
    
    
    def match(self, offense_elo, defense_elo, home):
        '''
        perform a match for the given year and week.  return the expected scores.
        '''
        # get home score which is matching the home offense to the away defense
        return (offense_elo - defense_elo) / self.div + self.points_offset
    
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
        self.update_team_offense_elo(offense, yearweek, oe + self.k_o * (score - expected))
        self.update_team_defense_elo(defense, yearweek, de + self.k_d * (expected - score))
    
    def __year_week_combine__(self, year, week):
        return year * 100 + week
    
    def load_all_matches(self, verbose=False):
        last_offense_elo = {}
        last_defense_elo = {}
        
        for year in range(START_YEAR, self.last_year_played + 1):
            print ('Parsing year %d' % year)
            for week in range(1, self.last_week_played + 1):
                yearweek = self.__year_week_combine__(year, week)
                for game in nflgame.games(year, week):
                    home, away = get_team(game.home), get_team(game.away)
                    for t in [home, away]:
                        if home not in last_offense_elo:
                            last_offense_elo[t] = STARTING_ELO
                            last_defense_elo[t] = STARTING_ELO
                    # update the elo based on the score of the home and away 
                    for t1, t2, score, homet in [(home, away, game.score_home, True), (away, home, game.score_away, False)]:
                        expected = self.match(self.get_offense_elo(t1, yearweek-1), self.get_defense_elo(t2, yearweek-1), homet)
                        self.update(yearweek, t1, t2, score, expected)
                        if verbose:
                            print ('home: %s away: %s score: %d expected %d' % (home, away, score, expected) )

                    
