import nflgame

def gen_sched(year):
    games = nflgame.sched.games
    return_games = {i:[] for i in range(1,18)}
    
    for gn, game  in games.items():
        
        yeari, week, home, away = int(game['year']), int(game['week']), game['home'], game['away']
        #print (week)
        if game['season_type'] == 'REG' and yeari == year:
            return_games[week].append((home, away))
    return return_games