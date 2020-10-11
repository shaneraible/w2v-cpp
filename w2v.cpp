#include "w2v.h"
#include <iostream>
#include <math.h>
#include <fstream>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40


clock_t start;

long long w2v::maxVocabSize, w2v::vocabSize, w2v::layer1Size, w2v::fileSize;
std::string w2v::trainFile;
int w2v::debugMode;
int w2v::numThreads;
w2v::real w2v::alpha, w2v::startingAlpha, w2v::subsample ;
w2v::real *w2v::hiddenLayer, *w2v::outLayerSoft, *w2v::outLayerNeg, *w2v::expTable;
long long trainWords = 0, wordCountActual = 0, iter = 30, fileSize = 0, classes = 0;
int w2v::vocabHashSize, w2v::numChars, w2v::tableSize;
int* w2v::vocabHash;
w2v::vocabWord* w2v::vocab;
unsigned int w2v::window;
int w2v::negative;
int *w2v::uniTable;

w2v::w2v(){
    vocabHashSize=30E6;
    numThreads = 1;
    numChars=257;
    maxVocabSize=1000;
    vocabSize=0;
    layer1Size=300;
    negative = 5;

    fileSize=0;
    tableSize=1e8;
    trainFile="test.txt";
    alpha = .025;
    startingAlpha = 0;
    subsample = 1e-3;
    window = 5;
    vocab = (struct vocabWord *)calloc(maxVocabSize, sizeof(struct vocabWord));
    vocabHash = (int *)calloc(vocabHashSize, sizeof(int));

    expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
    for (int i = 0; i < EXP_TABLE_SIZE; i++) {
        expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
        expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
    }

}

void w2v::trainModel(){
    long a, b, c, d;
    std::cout<<"Reading training file..."<<std::endl;
    learnVocabFromTrainFile();
    saveVocab("text_vocab.txt");
    std::cout<<"initializing network..."<<std::endl;
    initNet();
    std::cout<<"creating unigram table..."<<std::endl;
    initUnigramTable();
    std::cout<<"starting the clock..."<<std::endl;
    start = clock();
    std::cout<<"allocating the threads..."<<std::endl;
    pthread_t *pt = (pthread_t *)malloc(numThreads * sizeof(pthread_t));

    startingAlpha = alpha;
    std::cout<<"creating the threads..."<<std::endl;
    for (a = 0; a < numThreads; a++) pthread_create(&pt[a], NULL, trainModelThread, (void *)a);
    std::cout<<"joining the threads..."<<std::endl;
    for (a = 0; a < numThreads; a++) pthread_join(pt[a], NULL);

    for(a=0; a<layer1Size; a++)std::cout<<hiddenLayer[a]<<std::endl;
    
}

void *w2v::trainModelThread(void *id){
    std::cout<<"IN THE THREAD FOH TODAY: "<<(long long)id<<std::endl;
    unsigned long long next_random = (long long)id;
    long long wordCount=0, lastWordCount =0, word,  sen[120 + 1]; 
    long long sentencePos = 0, sentenceLength = 0, localIter = iter;
    unsigned long long nextRandom = (long long)id;
    char eof=0;

    clock_t now;
    real f, g;
    real *l1e = (real *)calloc(layer1Size, sizeof(real));
    std::fstream in (trainFile);
    
    while(1){
        std::cout<<localIter<<std::endl;
        if (wordCount - lastWordCount > 10) {
            wordCountActual += wordCount - lastWordCount;
            
            lastWordCount = wordCount;
            
            // The percentage complete is based on the total number of passes we are
            // doing and not just the current pass.      
            if ((debugMode > 1)) {
                now=clock();
                printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                // Percent complete = [# of input words processed] / 
                //                      ([# of passes] * [# of words in a pass])
                wordCountActual / (real)(iter * trainWords + 1) * 100,
                wordCountActual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                fflush(stdout);
            }
            alpha = startingAlpha*(1-wordCountActual/(real)(iter*trainWords + 1));
            if(alpha<startingAlpha*.0001) alpha = startingAlpha*.0001;
        }
        if (sentenceLength==0){
            std::string w;
            while(in>>w){ 
                word = searchVocab(w);
                wordCount++;
                if (word==-1) continue;
                if(word==0) break;

                if(subsample>0){
                    real ran = (sqrt(vocab[word].count / (subsample * trainWords)) + 1) * (subsample * trainWords) / vocab[word].count;
                    next_random = next_random * (unsigned long long)25214903917 + 11;
                    if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                }
                sen[sentenceLength++] = word;
                std::cout<<vocab[word].word<<std::endl;
                //TODO MAX SENTENCE LENGTH
            }

            sentencePos = 0;
        }

        if(in.get()==-1 || (wordCount>trainWords/numThreads)){
            wordCountActual +=wordCount-lastWordCount;
            localIter--;
            std::cout<<localIter<<std::endl;
            if(localIter==0) break;
            wordCount=0;
            lastWordCount=0;
            sentenceLength=0;
            //TODO FSEEK
            continue; 
        }

        word=sen[sentencePos];
        if(word == -1) continue;
        for(int c=0; c<layer1Size; c++) l1e[c]=0;
        nextRandom = nextRandom* (unsigned long long)25214903917 + 11;
        long long b = nextRandom%window;

        //TODO CBOW LATER
        if(1){
            for(int a=b; a<window*2+1-b; a++)  if(a!=window){
                int c = sentencePos - window + a;
                if(c<0) continue;
                if(c>=sentenceLength) continue;

                long long lastWord = sen[c];

                if(lastWord==-1) continue;

                long long l1 = lastWord*layer1Size;

                for(int c=0; c<layer1Size; c++) l1e[c]=0;

                if(negative>0){
                    long long target=-1;
                    int label=0;
                    for(int d=0; d<negative+1; d++){
                        if(d==0){
                            target = word;
                            label=1;
                        }else{
                            nextRandom = nextRandom*(unsigned long long)25214903917 + 11;
                            target = uniTable[(nextRandom>>16)%tableSize];

                            if(target==0) target = nextRandom % (vocabSize-1)+1;
                            if(target==word) continue;
                            label=0;
                            long long l2 = target*layer1Size;
                            long long f=0;
                            for(int c=0; c<layer1Size; c++)
                                f+=hiddenLayer[c+l1]*outLayerNeg[c+l2];
                            if(f>MAX_EXP)
                                g = (label-1)*alpha;
                            else if (f<-MAX_EXP)
                                g = label*alpha;
                            else
                                g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                            
                            for (int c = 0; c < layer1Size; c++) l1e[c] += g * outLayerNeg[c + l2];
                            for (int c = 0; c < layer1Size; c++) outLayerNeg[c + l2] += g * hiddenLayer[c + l1];
                        }
                    }
                }
                //TODO HIERARCHAL SOFTMAX

                for(int c=0; c<layer1Size; c++) hiddenLayer[c+l1] += l1e[c];

            }
        }
        sentencePos++;
        if(sentencePos>=sentenceLength){
            sentencePos=0;
            continue;
        }
        break;
    }

    in.close();
    free(l1e);
    pthread_exit(NULL); //returns
}

//TODO RENAME TO CREATE VOCAB FROM FILE!!! THIS IS NOT READING THE VOCAB
void w2v::learnVocabFromTrainFile(){
    for(int i=0; i<vocabHashSize; i++) vocabHash[i]=-1;

    std::string FILENAME = trainFile;
    std::ifstream infile(FILENAME);
    std::string sentence;
    
    addToVocab("</s>");

    if (infile.is_open()){
        while( getline (infile,sentence) ){
            std::string word ="";
            for(int i=0; i<sentence.size(); i++){
                if(sentence[i]!=' '){
                    word+=sentence[i];
                }
                if(sentence[i]==' '||i==sentence.size()-1){
                    int loc = searchVocab(word);
                    if(loc==-1){
                        int a = addToVocab(word);
                        vocab[a].count = 1;
                    }else{
                        vocab[loc].count++;
                    }
                    
                    word="";
                }                    
            }
        }
        infile.close();
    }
    //TODO NOT THIS OF COURSE
    FILE *a=fopen(trainFile.c_str(), "rb");
    fileSize = ftell(fopen(trainFile.c_str(), "rb"));
    fclose(a);
    sortVocab();
}

int w2v::VocabCompare(const void *a, const void *b) {
  long long l = ((struct vocabWord *)b)->count - ((struct vocabWord *)a)->count;
  if (l > 0) return 1;
  if (l < 0) return -1;
  return 0;
}

void w2v::sortVocab(){
    std::qsort(&vocab[1], vocabSize-1, sizeof(struct vocabWord), VocabCompare );
    for (int a = 0; a < vocabHashSize; a++) vocabHash[a] = -1;
    trainWords = 0; 
    int size = vocabSize;
    for (int a = 0; a < size; a++) {
        // // Words occuring less than min_count times will be discarded from the vocab
        // if ((vocab[a].count < minCount) && (a != 0)) {
        //     vocabSize--;
        //     free(vocab[a].word);
        // } else {
        // Hash will be re-computed, as after the sorting it is not actual
        int hash=getHash(vocab[a].word);
        while (vocabHash[hash] != -1) hash = (hash + 1) % vocabHashSize;
        vocabHash[hash] = a;
        trainWords += vocab[a].count;

        // }
    }
}

void w2v::saveVocab(std::string filename){
    std::ofstream out(filename);
    for(int i=0; i<vocabSize; i++) out<<vocab[i].word<<" "<<vocab[i].count<<std::endl;
    out.close();
}

void w2v::initNet(){
    long long a,b;
    unsigned long long nextRandom = 1;

    //allocating space for the hidden layer
    std::cout<< (long long)vocabSize*layer1Size*sizeof(real) <<std::endl;
    std::cout<< sizeof(void *) <<std::endl;
    
    a = posix_memalign((void **)&hiddenLayer, 128, (long long)vocabSize*layer1Size*sizeof(real));
    std::cout<< hiddenLayer <<std::endl;    
    
    if(hiddenLayer==NULL) {std::cout<<"Memory allocation failed at hidden layer\n"; exit(1);}
    //TODO Hierarchal softmax for training

    //negative sampling for trainign...
    bool negative=1;
    if(negative>0){
        //allocate space for the outputlayer of the model
        a = posix_memalign((void **) &outLayerNeg, 128, (long long)vocabSize*layer1Size*sizeof(real));
        
        if(outLayerNeg==NULL) {std::cout<<"Memory allocation failed at output layer\n"; exit(1);}
    }

    for(a=0; a<vocabSize; a++)
        for(b=0; b<layer1Size; b++){
            //TODO MAKE BETTER RANDOM NUMBER
            nextRandom = nextRandom * (unsigned long long)25214903917 + 11;
            hiddenLayer[a*layer1Size+b] = (((nextRandom&0xFFFF)/ (real)65536) - 0.5) / layer1Size;
        }

    //TODO CREATE BINARY TREE FOR HIERARCHICAL SOFTMAX
    createBinaryTree();
}
/*
 * This table is used for negative sampling
 * it will update x number of "negative" words 
 * with each added sentence instead of all words
 * x = 5-20 for small datasets, x = 2-5 for large
 * P(Wi) = f(wi)^.75/sum(f(wj)^.75)
 * This table holds the index of words in the vocab
 * with frequencies the same as the vocab*/
void w2v::initUnigramTable(){
    int a, i;
    double trainWordsPow = 0;
    double d1, power = 0.75;

    uniTable = (int *)malloc(tableSize * sizeof(int));
    
    //sum(f(wj)^.75): segfault if vocabSize = 0
    for (a = 0; a < vocabSize; a++) 
        trainWordsPow += pow(vocab[a].count, power);

    i = 0;
    d1 = pow(vocab[i].count, power) / trainWordsPow;

    for (a = 0; a < tableSize; a++) {
        uniTable[a] = i;
        if (a / (double)tableSize > d1) {
        i++;
        d1 += pow(vocab[i].count, power) / trainWordsPow;
        }
        if (i >= vocabSize) i = vocabSize - 1;
    }
}

int w2v::getHash(std::string word){
    unsigned long long a, hash = 0;
    for (a = 0; a < word.size(); a++) 
        hash = hash * (numChars + 1) + word[a];

    hash = hash % vocabHashSize;
    return hash;
}

int w2v::getWordIndex(std::string word){ 
    return -1; 
}  //index of word in vocab

int w2v::searchVocab(std::string word){
    unsigned int h = getHash(word);
    
    while(1){
        if(vocabHash[h]==-1) return -1;
        if(vocab[vocabHash[h]].word==word) return vocabHash[h];

        h = (h+1)%vocabHashSize;
    }

    return -1;
}

int w2v::addToVocab(std::string word){ 
    unsigned int hash, length = word.size();

    vocab[vocabSize].word = word;
    vocab[vocabSize].count = 0;
    vocabSize++;
    if (vocabSize + 2 >= maxVocabSize) {
        maxVocabSize += 1000;
        vocab = (struct vocabWord *)realloc(vocab, maxVocabSize * sizeof(struct vocabWord));
    }

    hash=getHash(word);
    while(vocabHash[hash] !=-1) hash = (hash + 1) % vocabHashSize;
    vocabHash[hash] = vocabSize - 1;
    return vocabSize-1;
}

void w2v::createBinaryTree (){

}