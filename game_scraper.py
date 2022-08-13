from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
from sportsreference.nfl.schedule import Schedule 
from sportsreference.nfl.teams import Teams
from datetime import datetime 
import multiprocessing as mp
import os, pickle, shelve
lock = mp.Lock()
lock_load = mp.Lock()
START_YEAR = 2009

class Game:
            def __init__(self, game):
                self.keys = game.keys()
                self.__dict__.update( game)
            
            def __repr__(self):
                return str(self.__dict__)
            
class game_scraper():
    reload = False
    
    def __init__(self, to_year= 2019):
        self.cache_filename = self.__class__.__name__ + '.db'

        self.load()
        if game_scraper.reload:
            self.clear()
        
        self._teams = self.cache['teams'] if 'teams' in self.cache else set([])
        self._games = self.cache['games']
        last_year = self.cache['last_year']

        #print ('last year ' + str(last_year))
        
        for year in range(START_YEAR, to_year + 1):
            if year <= last_year and not game_scraper.reload:
                continue
            print (year)
            year_done = True
            for team in Teams(year):
                self._teams.add(team.abbreviation)
                s = Schedule(team.abbreviation, year)
                print (s.dataframe)
                for week, row in enumerate(s.dataframe.iterrows()):
                    if row[1]['datetime'] > datetime.now():
                        year_done = False
                        
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
        self.teams = [[team] for team in self._teams]
        #if self._games != self.cache['games']:
        self.cache['games'] = self._games
        self.cache['teams'] = self._teams
        self.cache.sync()
        
        games = {}
        for game in self._games:
            games[game] = [Game(g2) for g2 in  self._games[game].values()]
        self._games = games
        self.games(2019, 1)
        print(str(self.teams))
        assert(len(self.teams) == 32)
        game_scraper.reload=False
    
    def clear(self):
        self.cache['games'] = {}
        self.cache['last_year'] = 0
        self.cache['teams'] = set([])
    
    
    def load(self):
        self.cache = shelve.open(self.cache_filename, writeback = True)
        #if 'games' not in self.cache:
        #    self.clear()

    def games(self, year, week):
        if (year, week) not in self._games:
            #assert (False)
            return []
        games = self._games[(year, week)]
        return games

if __name__ == '__main__':  
    game_scraper()