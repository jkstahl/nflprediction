from elo_model import elo_model
from game_scraper import game_scraper 
nflgame = game_scraper()

e= elo_model(2019, 17)
e.load_all_matches()
print (e.elo_offense)
print (e.elo_defense)

def simulate():
    total = 0.0
    correct = 0.0
    for year in range(2011, 2019):
        for week in range(1, 17+1):
            games = nflgame.games(year, week)
            for game in games:
                home_score, away_score = e.play_match(year, week, game.home, game.away)
                win_actual = game.score_home > game.score_away
                win_expect = home_score > away_score
                total += 1
                if win_actual == win_expect:
                    correct += 1 
                print ('home r/e: %d/%d away r/e: %d/%d' % ( game.score_home, home_score, game.score_away, away_score))
    print (correct / total)

simulate()