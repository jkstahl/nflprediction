import nflgame
import numpy as np


def main():
    
    home_stats = []
    for team in nflgame.teams:
        try:
            home_x = []
            home_y = []
            away_x = []
            away_y = []
            for year in range(2009, 2017):
                games = nflgame.games(year, home=team[0])
                
                for game in games:
                    home_x.append(game.score_home)
                    home_y.append(game.score_away)
                games = nflgame.games(year, away=team[0])
                for game in games:
                    away_x.append(game.score_away)
                    away_y.append(game.score_home)    
            home_x = np.array(home_x)
            home_y = np.array(home_y)
            away_x = np.array(away_x)
            away_y = np.array(away_y)
            affect_off = (home_x.mean() - away_x.mean()) / away_x.mean() * 100
            affect_def = (away_y.mean() - home_y.mean()) / away_y.mean() * 100
            home_stats.append((affect_off, affect_def, team[0]))
            #print ('%s\t OFFENSE:%.1f%%,\t DEFENSE:%.1f%%' % (team[0], affect_off, affect_def))
        except:
            pass
    
    for affect_off, affect_def, team in sorted(home_stats, reverse=True):
        print ('%s\t OFFENSE:%.1f%%,\t DEFENSE:%.1f%%' % (team, affect_off, affect_def))
    
    home_x = []
    home_y = []
    away_x = []
    away_y = []
    for year in range(2009, 2017):
        games = nflgame.games(year, home='NO')
        for game in games:
            home_x.append(game.score_home)
            home_y.append(game.score_away)
        games = nflgame.games(year, away='NO')
        for game in games:
            away_x.append(game.score_away)
            away_y.append(game.score_home)
    
    import matplotlib.pyplot as plt
    home_x = np.array(home_x)
    home_y = np.array(home_y)
    away_x = np.array(away_x)
    away_y = np.array(away_y)
    plt.plot(home_x, home_y, 'o', away_x, away_y, 'x', home_x.mean(), home_y.mean(), 'or', away_x.mean(), away_y.mean(), 'oy')
    plt.plot([0,50], [0,50], 'r-')
    plt.show()
        
    
if __name__ ==  '__main__':
    main()