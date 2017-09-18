import nflgame
import numpy as np
import random
import matplotlib.pyplot as plt
from statistical_model import home_model as team_tracker 
from modeler import get_team
#from prediction_model_ai import prediction_model_ai as team_tracker
import threading
import cProfile
#from prediction_model_ai import prediction_model_ai
from optparse import OptionParser


#prediction_model_ai(10)

from multiprocessing.dummy import Pool 

DEBUG = True
np.random.seed(2)

SIMS = 1000

games_cache = {}


def eval_prediction(game, prediction_home, prediction_away, winner):
    actual_winner = get_team(game.winner)
    correct = actual_winner==winner
    if DEBUG:
        print 'PREDICTED: %s ACTUAL %s - %s' % (winner, actual_winner,correct)
    return (((game.score_home-prediction_home)**2)+((game.score_away-prediction_away)**2), correct)


def eval_games(games, predictions):
    error_sum = 0
    correct_sum = 0
    for game in games:
        
        home, away, winner = predictions[get_team(game.home)]
        error, correct = eval_prediction(game, home, away, winner)
        error_sum += error
        correct_sum += int(correct)
    return (1.0 * error_sum / len(predictions)/2, 1.0 * correct_sum / len(predictions))


def predict_week(season, week, go_back, print_it = False, team_stats = None):
    print 'season: %d, week: %d' % (season, week)
    if team_stats == None:
        team_stats = team_tracker()
    # get history up to and including week - go_back
    week_delta = (week - go_back)
    from_year = season - abs(week_delta) / 17 
    if week_delta <= 0:
        from_year -= 1
    from_week = ((week_delta - 1) % 17) + 1
    to_year = season if week > 1 else season - 1
    to_week = week - 1 if week > 1 else 17
    if print_it:
        print ('%d-%d %d-%d' % (from_year, to_year, from_week, to_week))
    team_stats.process_history(from_year, to_year, from_week, to_week)
    
    
    # simulate all games of a given week with the
    predictions = {} 
    
    year = season
    for game in nflgame.live._games_in_week(year, week):
        home = get_team(game['home'])
        away = get_team(game['away'])
        home_ave, away_ave, home_prob, away_prob = team_stats.play_match(home, away, None, None, SIMS)
        predictions[home]=(home_ave, away_ave, home if home_prob >= .5 else away)
        if print_it:
            print '%s-%d, %s-%d, %s-%.1f%%, %s-%.1f%%' % (home, home_ave, away, away_ave, home, home_prob * 100, away, away_prob* 100)
    
    #print (str(team_stats.play_match('SEA', 'GB', None, None)))
    if (year, week) in games_cache:
        games = games_cache[(year, week)]
    else:
        games = nflgame.games(year, week)
        games_cache[(year, week)] = games
    
        
    return eval_games(games, predictions)
    
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
    parser.add_option('-g', '--go_back', dest='go_back', type='int', help='The amount of time (in NFL weeks) to accumulate stats for the prediction model 15-19 is good.', default=19)
    parser.add_option("-s", "--simulation", action="store_true", dest="simulation", default=False, help="Run back testing simulation.")    
    (options, args) = parser.parse_args()

    '''
    team_stats = team_tracker()
    team_stats.process_history(2013, 2013, 1, 17)
    print (str(team_stats.play_match('SEA','GB', None, None, 10000)))
    return 0 
    '''
    
    if options.predict_week != None:
        year, week = options.predict_week.split('-')
        year, week = int(year) , int(week)
        print ('Predicting all games for year: %d, week %d' % (year, week))
        error, accuacy = predict_week(year, week, options.go_back, True)
        print ('Results: error %.2f, accuracy %.3f' % (error, accuacy))
    
    if options.simulation:    
        go_backs = range(4, 20, 2)
        acs = []
        weeks_acc = np.array([0.0] * 17)
        dates_acc = np.array([0.0] * (END_YEAR-START_YEAR + 1)*WEEKS_PER_SEASON)
        
        dates_total = np.array([0.0] * (END_YEAR-START_YEAR + 1)*WEEKS_PER_SEASON)
        weeks_total = np.array([0.0] * 17)
        
        model = team_tracker()
        
        for go_back in go_backs:
            week_range = (END_YEAR-START_YEAR + 1)*WEEKS_PER_SEASON - go_back
            num_tests = TESTS if RANDOM else week_range 
            print ('Number of weeks of sim %d' % week_range)
            accuracy = 0
            j = 1
            
            for i in range(num_tests):
                if RANDOM:
                    raw_week = random.randrange(0, week_range) + 1
                else:
                    raw_week = j
                    j+=1
                year = START_YEAR + ((raw_week + go_back) / WEEKS_PER_SEASON)
                week = (raw_week + go_back) % WEEKS_PER_SEASON + 1
                
                #print ('Year: %d, Week: %d' % (year, week))
                error, percentage = predict_week(year, week, go_back, False, model)
                #error, percentage = predict_week(year, week, go_back)
                print ('Error: %.2f, Percentage: %.2f' % (error, percentage))
                accuracy += percentage
                weeks_acc[week-1] += percentage
                weeks_total[week-1] += 1
                
                dates_acc[raw_week-1 + go_back] += percentage
                dates_total[raw_week-1 + go_back] += 1
                
                i+=1
            ave_ac = accuracy / num_tests
            acs.append(ave_ac)
            print ('Total: %.2f' % (ave_ac))
        
        print ('Accuracy by week:') 
        plt.figure(1)
        plt.title('Accuracy by week')
        week_accuracy = weeks_acc/weeks_total
        plt.plot(week_accuracy)   
        print (str(weeks_acc/weeks_total))
        print ('total accuracy: %s' %str(acs))
        plt.figure(2)
        plt.title('Accuracy by go back')
        plt.plot(go_backs, acs)
        
        plt.figure(3)
        plt.title('Date accuracy')
        date_acc_ave = dates_acc / dates_total
        plt.plot(date_acc_ave)
        plt.show()
    
if __name__ ==  '__main__':
    main()
    #cProfile.run('main()', 'simstats')
    #import pstats
    #p = pstats.Stats('simstats')
    #p.sort_stats('cumulative').print_stats(100)
    