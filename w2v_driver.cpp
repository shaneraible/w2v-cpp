#include "w2v.h"
#include <iostream>


int main(){
    std::cout<<"hello"<<std::endl;
    w2v words;
    words.trainModel();
    std::cout<<"Most similar to king: "<<words.getMostSimilar("king")<<std::endl;
    std::cout<<"Most similar to queen: "<<words.getMostSimilar("queen")<<std::endl;
    std::cout<<"Most similar to prince: "<<words.getMostSimilar("prince")<<std::endl;
    std::cout<<"Most similar to princess: "<<words.getMostSimilar("princess")<<std::endl;
    std::cout<<"Most similar to crown: "<<words.getMostSimilar("crown")<<std::endl;
}