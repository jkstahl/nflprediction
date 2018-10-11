import nflgame
import numpy as np
import random
import matplotlib.pyplot as plt
from statistical_model import home_model as stat_model 
from ai_model2 import new_model as nm, new_model
from modeler import get_team
import threading
import cProfile
#from prediction_model_ai import prediction_model_ai
from optparse import OptionParser
from multiprocessing.dummy import Pool 


np.random.seed(2)
SIMS = 100000
games_cache = {}

DEBUG = False


def get_games(year, week):
    return [('ATL', 'PHI'),
            ('BUF', 'BAL'),
            ('PIT', 'CLE'),
            ('CIN', 'IND'),
            ('TEN', 'MIA'),
            ('SF', 'MIN'),
            ('HOU', 'NE'),
            ('TB', 'NO'),
            ('JAX', 'NYG'),
            ('KC', 'LAC'),
            ('WAS', 'ARI'),
            ('DAL', 'CAR'),
            ('SEA', 'DEN'),
            ('CHI',  'GB'),   
            ('NYJ', 'DET'),
            ('LA',  'OAK'),]  


def eval_prediction(game, prediction_home, prediction_away, winner, print_it=False):
    actual_winner = get_team(game.winner)
    correct = actual_winner==winner
    if print_it:
        print 'PREDICTED: %s ACTUAL %s - %s' % (winner, actual_winner,correct)
    return (((game.score_home-prediction_home)**2)+((game.score_away-prediction_away)**2), correct)


def eval_games(games, predictions, print_it=False):
    if print_it:
        print ('Results:')
    error_sum = 0
    correct_sum = 0
    for game in games:
        home, away, winner = predictions[get_team(game.home)]
        error, correct = eval_prediction(game, home, away, winner, print_it)
        error_sum += error
        correct_sum += int(correct)
    return (1.0 * error_sum / len(predictions)/2, 1.0 * correct_sum / len(predictions))


def predict_week(year, week, alpha, season_discount, model, print_it = False):
    print 'Getting predictions for: year %d, week %d' % (year, week)
        
    # simulate all games of a given week with the
    predictions = {} 
    model.load_stats()
    model.process_records(alpha, season_discount)
    model.train(year, week)
    
    for game in nflgame.live._games_in_week(year, week):
        home = get_team(game['home'])
        away = get_team(game['away'])
        away_prob, home_prob, predict = model.predict(home, away)
        predictions[home]=(home if predict == 1 else away)
        if print_it:
            print '%s %f -%s %f    -> %s ' % (home, home_prob, away, away_prob, (away, home)[int(predict)])
            #print '%s: %s' % (home, zip(model.stats, model.get_s_vect(home).values))
            #print '%s: %s' % (away, zip(model.stats, model.get_s_vect(away).values))
    
    #print (str(team_stats.play_match('SEA', 'GB', None, None)))
    #games = nflgame.games(year, week)
    
    #return eval_games(games, predictions, print_it)
    
def prediction_wrapper(args):
    return predict_week(*args)

def main():
    #predict_week(2016, 1, 3)
    #predict_week(2016, 17, 17)
    START_YEAR = 2011
    END_YEAR = 2016
    WEEKS_PER_SEASON = 17
    TESTS = 20
    RANDOM = False

    parser = OptionParser()
    parser.add_option("-p", "--predict_week", dest="predict_week", help="Predict all games for the given week in the form. year-week", default=None)
    parser.add_option('-a', '--alpha', dest='alpha', type='float', help='Alpha to use in rolling average filter for stats', default=0.172)
    parser.add_option('-d', '--season_discount', dest='season_discount', type='float', help='Percentage to discount the stats from the previous season', default=0.75)
    parser.add_option("-s", "--simulation", action="store_true", dest="simulation", default=False, help="Run back testing simulation.")    
    parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="Run back testing simulation.")  
    (options, args) = parser.parse_args()

    '''
    team_stats = team_tracker()
    team_stats.process_history(2013, 2013, 1, 17)
    print (str(team_stats.play_match('SEA','GB', None, None, 10000)))
    return 0 
    '''
    model = new_model()
    if options.predict_week != None:
        year, week = options.predict_week.split('-')
        year, week = int(year) , int(week)
        print ('Predicting all games for year: %d, week %d' % (year, week))
        predict_week(year, week, options.alpha, options.season_discount, model,True)
        #print ('Results: error %.2f, accuracy %.3f' % (error, accuacy))
    
    #if options.simulation:    
        
    
if __name__ ==  '__main__':
    main()
    #cProfile.run('main()', 'simstats')
    #import pstats
    #p = pstats.Stats('simstats')
    #p.sort_stats('cumulative').print_stats(100)
    