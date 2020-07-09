cd ../covid19MLPredictor
eval "$(conda shell.bash hook)"
conda activate covid19
python pages/covid_parser.py
cd ../covid19_inference/scripts
python estimate_r_t.py
echo "Pushing new entries"
cd ../../covid19MLPredictor
git add *
git commit -m "daily updates"
git push heroku master
echo "Done new entries"
# cd ../covid19_inference/scripts
# cd scripts
# python example_qatar.py
# cd ../../covid19MLPredictor/
# git add * 
# git commit -m "daily updates"
# git push heroku master
# echo "Done"
