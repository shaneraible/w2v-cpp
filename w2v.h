#ifndef W2V_H
#define W2V_H

#include <string>

class w2v{
    typedef float real; //layer precision//float precision

    struct vocabWord {
        long long count;
        int *point;
        char *code, codelen;
        std::string word;
    };
    
    public:
        w2v();
        void trainModel();

    private:
        int vocabHashSize;
        int tableSize;
        int numChars;
        static long long maxVocabSize, vocabSize, layer1Size, fileSize;
        static std::string trainFile;
        struct vocabWord *vocab;
        int *uniTable;
        int *vocabHash;
        static int numThreads;
        static real alpha, startingAlpha, subsample;
        static int debugMode;
        
        real *hiddenLayer, *outLayerSoft, *outLayerNeg, *expTable;
        static int VocabCompare(const void *, const void *);
        void learnVocabFromTrainFile();
        void saveVocab(std::string);
        static void *trainModelThread(void *id);
        void initNet();
        void createBinaryTree();
        void sortVocab();
        int searchVocab(std::string word);
        int getHash(std::string word);       //hash value for a word
        int getWordIndex(std::string word);  //index of word in vocab
        int addToVocab(std::string word);
        void initUnigramTable();
};


#endif