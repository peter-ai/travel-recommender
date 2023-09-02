# Travel Recommendation System
**travel-recommender** is a script that acts a recommendation system which, based on your past experiences, provides useful suggestions for your next travel destination. Decisions can be informed by up to 3 past travel experiences; you can query the system for similar locations to those provided, or dissimilar destinations if you're looking for a change, outputting up to 25 suggestions for you to consider. 

Data underpinning these recommendations was scraped from Wikipedia, considering various characteristics of a destination inclduing climate, transportation modalities, demographics, food, political atmosphere, and geograpphical features, among others.

Requirements: *Python 3.11+*

## Installation
### Source code
The system can be cloned from the latest sources using this command: 
```
git clone https://github.com/peter-ai/travel-recommender.git
```

### Dependencies
THere are a number of python packages on which the system depends. Refer to the requirements.txt for specifics; to install necessary packages, use:
```
pip install -r requirements.txt
```

As of this release, the ```lightgbm``` package cannot be installed on Apple Silicon via pip. Alternatively, conda can be used or the package can be downloaded and built from the latest source package (see [LightGBM](https://pypi.org/project/lightgbm/3.3.4/)) by extracting the lightgbm-3.3.4 folder from the tar.gz and running: 
```
python setup.py install
```

Ensure you have ```cmake```, ```libomp```, and ```gcc``` or ```clang``` installed before executing the above. 

## Usage
1. Run `python travel_recommender.py` in your terminal, text editor or IDE
2. Enter 3 countries, sequentially, hitting enter after each entry (spelling need not be 100% correct)
3. Decide whether you want similar destinations to those you provided or not
4. Enter the number of recommendations you would like to recieve
5. Voila! A list of recommendations is generated and a visualization of them is presented

### CLI
[![asciicast](https://asciinema.org/a/KJpXLor4YrJ9EFrWccM90PNOL.svg)](https://asciinema.org/a/KJpXLor4YrJ9EFrWccM90PNOL?t=03&loop=0)

### Choropleth: Visualization of recommended travel destinations
![Choropleth exemplar](choropleth-example.png?raw=true "Choropleth: Visualization of recommended travel destinations")

## Acknowledgment
This project was inspired by my love for travel, and my parents, who's lives are only now allowing them to explore their love for it too.

## License
This project is licensed under [MIT license](http://opensource.org/licenses/MIT). For the full text of the license, see the [LICENSE](https://github.com/peter-ai/travel-recommender/blob/main/LICENSE) file.