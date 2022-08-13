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
START_YEAR = 2010

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
        self._stats = self.cache['stats']
        self._cached_tuples = self.cache['_cached_tuples']
        last_year = self.cache['last_year']

        #print ('last year ' + str(last_year))
        
        for year in range(START_YEAR, to_year + 1):
            if year in self._cached_tuples and not game_scraper.reload:
                continue
            self._cached_tuples.add(year)
            print (year)
            year_done = True
            for team in Teams(year):
                self._teams.add(team.abbreviation)
                s = Schedule(team.abbreviation, year)
                print (s.dataframe)
                if team.abbreviation in self._stats:
                        self._stats[team.abbreviation] = self._stats[team.abbreviation].append(s.dataframe)

                else:
                    self._stats[team.abbreviation] = s.dataframe
                
            if year_done:
                self.cache['last_year'] = year
        
        #print (self._stats)
        
        #print (self._games[(2020,1)])
        self.teams = [[team] for team in self._teams]
        #if self._games != self.cache['games']:
        self.cache['stats'] = self._stats
        self.cache['teams'] = self._teams
        self.cache['_cached_tuples'] = self._cached_tuples
        self.cache.sync()
        
        print(str(self.teams))
        assert(len(self.teams) == 32)
        game_scraper.reload=False
    
    def clear(self):
        self.cache['stats'] = {}
        self.cache['last_year'] = 0
        self.cache['teams'] = set([])
        self.cache['_cached_tuples'] = set([])
    
    
    def load(self):
        self.cache = shelve.open(self.cache_filename, writeback = True)


    def stats(self):
        return self._stats
        

if __name__ == '__main__':  
    gs = game_scraper()