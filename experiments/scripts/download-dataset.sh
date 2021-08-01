mkdir -p dataset

wget http://www.tkl.iis.u-tokyo.ac.jp/~ishiwatari/naacl_data.zip
unzip naacl_data.zip
mv ./data ./dataset/ishiwatari
rm naacl_data.zip

# STS2012
mkdir -p dataset/sts/2012
wget http://ixa2.si.ehu.es/stswiki/images/4/40/STS2012-en-test.zip
unzip STS2012-en-test.zip
mv test-gold dataset/sts/2012/test
rm STS2012-en-test.zip

# STS2013
mkdir -p dataset/sts/2013
wget http://ixa2.si.ehu.es/stswiki/images/2/2f/STS2013-en-test.zip
unzip STS2013-en-test.zip
mv test-gs dataset/sts/2013/test
rm STS2013-en-test.zip

# STS2014
mkdir -p dataset/sts/2014
wget http://ixa2.si.ehu.es/stswiki/images/8/8c/STS2014-en-test.zip
unzip STS2014-en-test.zip
mv sts-en-test-gs-2014 dataset/sts/2014/test
rm STS2014-en-test.zip

# STS2015
mkdir -p dataset/sts/2015
wget http://ixa2.si.ehu.es/stswiki/images/d/da/STS2015-en-test.zip
unzip STS2015-en-test.zip
mv test_evaluation_task2a dataset/sts/2015/test
rm STS2015-en-test.zip

# STS2016
mkdir -p dataset/sts/2016
wget http://ixa2.si.ehu.es/stswiki/images/9/98/STS2016-en-test.zip
unzip STS2016-en-test.zip
mv sts2016-english-with-gs-v1.0 dataset/sts/2016/test
rm STS2016-en-test.zip

# STS2017
mkdir -p dataset/sts/2017
wget http://ixa2.si.ehu.es/stswiki/images/2/20/Sts2017.eval.v1.1.zip
unzip Sts2017.eval.v1.1.zip
wget http://ixa2.si.ehu.es/stswiki/images/7/70/Sts2017.gs.zip
unzip Sts2017.gs.zip
rm Sts2017.eval.v1.1.zip Sts2017.gs.zip
mv STS2017.eval.v1.1 dataset/sts/2017/input
mv STS2017.gs dataset/sts/2017/gs


# STS Benchmark
wget http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz
tar -zxvf Stsbenchmark.tar.gz
mv stsbenchmark dataset/sts/
rm Stsbenchmark.tar.gz


# SICK
wget http://alt.qcri.org/semeval2014/task1/data/uploads/sick_test_annotated.zip
unzip sick_test_annotated.zip -d SICK
mv SICK dataset/sts/
rm sick_test_annotated.zip