from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from sportsreference.nfl.schedule import Schedule 
from sportsreference.nfl.teams import Teams
from datetime import datetime 
import os, pickle, shelve
START_YEAR = 2009

class Game:
            def __init__(self, game):
                self.keys = game.keys()
                self.__dict__.update( game)
            
            def __repr__(self):
                return str(self.__dict__)
            
class game_scraper():

    
    def __init__(self):
        self.cache_filename = self.__class__.__name__ + '.db'
        self.load()
        reload = False
        
        if reload:
            self.clear()
        
        self._teams = self.cache['teams'] if 'teams' in self.cache else []
        self._games = self.cache['games']
        last_year = self.cache['last_year']
        print ('last year ' + str(last_year))
        
        for year in range(START_YEAR, 2020):
            if year <= last_year and not reload:
                continue
            print (year)
            year_done = True
            for team in Teams(year):
                self._teams.append(team.abbreviation)
                s = Schedule(team.abbreviation, year)
                for week, row in enumerate(s.dataframe.iterrows()):
                    if row[1]['datetime'] > datetime.now():
                        year_done = False
                        break
                    ha = row[1]['location'].lower()
                    yw = (year, row[1].week)
                    if yw not in self._games:
                        self._games[yw] = {}
                    if row[1]['boxscore_index'] not in self._games[yw]:
                        self._games[yw][row[1]['boxscore_index']] = {}
                    self._games[yw][row[1]['boxscore_index']].update({
                                                                            '%s' % ha : team.abbreviation ,
                                                                            'score_%s' % ha : row[1]['points_scored']
                                                                            }) 
            if year_done:
                self.cache['last_year'] = year
        
        #print (self._games[(2020,1)])
        self._teams = set(self._teams)
        print (self._teams)
        self.teams = [[team] for team in self._teams]
        self.cache['games'] = self._games
        self.cache['teams'] = self._teams
        self.cache.sync()
        games = {}
        for game in self._games:
            games[game] = [Game(g2) for g2 in  self._games[game].values()]
        self._games = games
        print (self.games(2019, 1))
        print (len(self.teams))
    
    def clear(self):
        self.cache['games'] = {}
        self.cache['last_year'] = 0
        self.cache['teams'] = []
    
    
    def load(self):
        self.cache = shelve.open(self.cache_filename, writeback = True)
        if 'games' not in self.cache:
            self.clear()

    def games(self, year, week):

        return self._games[(year, week)]
        
game_scraper()