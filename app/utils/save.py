import csv
import os

class Save:
    def create(filename, result):
        dir = 'results'
        # Check if directory exists
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        with open(f'{dir}/{filename}.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(result[:5])