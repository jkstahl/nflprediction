

print ('got to here')
from elo_model import elo_model
from bs4 import BeautifulSoup
import urllib.request


from game_scraper import game_scraper 
from multiprocessing import Process
nflgame = game_scraper(2021)
import numpy as np
import multiprocessing as mp
import random
lock = mp.Lock()

class Game:
    def __init__(self, home, away):
        self.home = home 
        self.away = away
        self.score_home = 0
        self.score_away = 0
        
        
def get_schedule(year, week):
    translate = {'TB':'TAM', 'SF':'SFO', 'TEN' : 'OTI', 'ARI' : 'CRD', 'IND' : 'CLT', 'WSH' : 'WAS', 'LAC': 'SDG', 'HOU': 'HTX', 'KC':'KAN', 'NE':'NWE', 'LV': 'RAI', 'NO': 'NOR', 'GB': 'GNB', 'LAR':'RAM', 'BAL':'RAV'}
    page = urllib.request.urlopen('https://www.espn.com/nfl/schedule/_/week/%d/year/%d' % (week, year))
    soup = BeautifulSoup(page.read(), 'html.parser')
    matches = []
    for sched in soup.find_all('table', class_='schedule'):
        for matchup in sched.find_all('tr'):
            matchup = matchup.find_all('abbr')
            m = [translate.get(team.text, team.text) for team in matchup]
            if len(m) == 2:
                matches.append(Game(m[1], m[0]))
    return matches

def get_manual_games():
    games = [
                ('BUF', 'MIA'),
                ('CIN', 'RAV' ),
                ('TAM', 'ATL'),
                ('NYG', 'DAL'),
                ('NWE', 'NYJ'),
                ('DET', 'MIN'),
                ('CLE', 'PIT'),
                ('SFO', 'SEA'),
                ('RAM', 'CAR'),
                ('CLT', 'JAX'),
                ('HTX', 'OTI'),
                ('DEN', 'RAI'),
                ('KAN', 'SDG'),
                ('CHI','GNB'),
                ('CAR', 'NOR'),
                ('PHI', 'WAS')
            ]
    return [Game(home, away) for home, away in games]
    
def get_predictions(year, week, params):
    e = elo_model(2021, 17, nflgame)
    e.set_params(params)
    e.load_all_matches()
    
    games = nflgame.games(year, week)
    if len(games) == 0: games = get_schedule(year, week)
    correct = 0
    total = 0
    for game in games:
        home_score, away_score = e.play_match(year, week, game.home, game.away)
        #win_actual = game.score_home > game.score_away
        win_expect = home_score > away_score

        
        print ('home %s  away %s  home score: %f  away score: %f  diff: %d  outcome: %s ' % (game.home, game.away, home_score, away_score, round(home_score-away_score), 'WIN' if win_expect else 'LOSE' ))
        if game.score_home == None:
            continue
        #print ('                  home score: %d  away score: %d' % (game.score_home, game.score_away))
        win_actual = game.score_home > game.score_away
        correct += int(win_actual == win_expect)
        total += 1
        if win_actual == win_expect: print ("CORRECT")
        else: print ('WRONG')
    print (1.0 * correct / total)

def simulate(params = (4, 4, 25, 25)):
    with lock:
        e = elo_model(2019, 17, nflgame)
        e.set_params(params)
        e.load_all_matches()
    
    total = 0.0
    correct = 0.0
    for year in range(2011, 2019):
        for week in range(1, 17+1):
            with lock:
                games = nflgame.games(year, week)
            for game in games:
                with lock:
                    home_score, away_score = e.play_match(year, week, game.home, game.away)
                win_actual = game.score_home > game.score_away
                win_expect = home_score > away_score
                total += 1
                if win_actual == win_expect:
                    correct += 1 
                #print ('home r/e: %d/%d away r/e: %d/%d' % ( game.score_home, home_score, game.score_away, away_score))
    return correct / total

def genetic():
    mins = np.array([[.1, .1, .1, .1]])
    maxs = np.array([[20, 20, 100, 100]])
    POP_SIZE = 32
    SURVIVE = .2
    MUTATE = .1
    rounds = 5
    population = np.random.random(size=(POP_SIZE, mins.size)) * (np.repeat(maxs, POP_SIZE, 0) - np.repeat(mins, POP_SIZE, 0)) + np.repeat(mins, POP_SIZE, 0)
    for r in range(rounds):
        print ("Round: %d" % (r + 1))
        
        # evaluate
        pool = mp.Pool(processes=8)
        result_list = np.array(pool.map(simulate, population))
        # sort in decending order
        sorted_ind = np.argsort(result_list)[::-1]
        population = population[sorted_ind]
        result_list = result_list[sorted_ind]
        print (result_list)
        
        # kill get n top candidates
        num_cadidates = int(POP_SIZE * SURVIVE)
        #population = population[:num_cadidates] # kills of bad candidates

        # breed: create new subset of breeders. sometimes they may breed with themselves which is ok.
        new_pop = POP_SIZE - num_cadidates
        males = np.random.binomial([POP_SIZE] * new_pop, [.3] * new_pop)
        #males = np.random.randint(POP_SIZE, size = new_pop)
        #females = np.random.randint(num_cadidates, size = new_pop)
        females = []
        for m in males:
            cand = list(range(POP_SIZE))
            cand.pop(cand.index(m))
            females.append(random.choice(cand))
        females = np.array(females)
        print ('males:   %s' % str(males))
        print ('females: %s' % str(females))
        offspring = (population[males] + population[females]) / 2 # offspring is the average
        population = np.append(population[:num_cadidates], offspring,0)
        print (population)
        print (population.shape)
        
        # mutate: get small subset and alter a little.  NOt the top subset
        mut_num = max(int(MUTATE * POP_SIZE), 1)
        mutation = np.random.normal(size=(mut_num, population.shape[1]))
        population[POP_SIZE-mut_num:POP_SIZE] = (population[POP_SIZE-mut_num:POP_SIZE] + population[POP_SIZE-mut_num:POP_SIZE] * mutation) 

    print (population[0])

#simulate()
def main():
    #genetic()
    #print (simulate([ 8.45917724,  6.73694828, 19.23659726, 44.69264598]))
    
    get_predictions(2021, 18,[  5.35392252,   4.4185035,   48.37095529 , 76.93300668])
    
if __name__ ==  '__main__':
    main()