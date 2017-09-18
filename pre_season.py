from sim import team_tracker as team_tracker
import numpy as np
import matplotlib.pyplot as plt

FROM_YEAR = 2009
TO_YEAR = 2016


def main():
    
    stats = ['yards_per_pass', 'yards_per_rush', 'points_per_yard']
    cors = []
    for i, stat in enumerate(stats):
        reg_aves = []
        pre_aves = []
        reg_prev = {}
        pre_prev = {}
        for year in range(FROM_YEAR, TO_YEAR + 1):
            
            # regular year
            tt = team_tracker()
            tt.process_history(year, year, 1, 17)
            teams = tt.get_teams()[:]
            for team in teams:
                current = tt.get_average(team, stat)
                reg_aves.append(current)
                
            
            # pre season]
            # regular year
            tt = team_tracker()
            tt.process_history(year, year, 1, 4, week_type = 'PRE')
            for team in teams:
                current = tt.get_average(team, stat)
                pre_aves.append(current)       
        print (str(reg_aves))
        print (str(pre_aves))
        
        # normalize by average
        reg_aves = np.array(reg_aves)
        pre_aves = np.array(pre_aves)
        
        pre_aves = pre_aves / pre_aves.mean() - 1
        reg_aves = reg_aves / reg_aves.mean() - 1
        
        plt.subplot(1, 3, i+1)
        plt.plot(pre_aves, reg_aves, '+')
        plt.xlabel("Regular Season Delta")
        plt.ylabel('Pre-season Delta')
        plt.title(stat)
        cors.append(np.corrcoef(pre_aves, reg_aves)[0, 1])
    
    print (str(cors))    
    plt.show()
    
    

if __name__ ==  '__main__':
    main()