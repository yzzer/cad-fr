## 人脸签到系统
> [front end](https://github.com/BUPT-CAD-Lab/annual-metting-fe)
## install
1. clone
```
git clone https://github.com/yzzer/cad-fr.git
```
2. install deepface
```
cd cad-fr/deepface
pip install -e .
```
3. install requirements
```
pip install -r requirements.txt
```

## run
0. modify config (`config/config.yaml`) and env (`bin/env.sh`)
1. create db
```
bin/create_db.sh
```

2. parse deepface db to sqlite
```
bin/parse.sh parse
```

3. clear checkin status
```
bin/parse.sh clear
```

4. run

