# KMeansProofOfConcept

This is a simple proof of concept for k-means clustering implemented in Python. Though numerous effective libraries exist for k-means,this was built from the ground up (though I do use Numpy to speed up matrix operations), for the purposes of exposing myself to the inner workings of k-means. Ultimately, I would like to use this to build an image compression k-means algorithm.

IMAGE COMPRESSION:
It is possible to use this to compress images by limiting to a certain numer of colors ("clusters" here). Implementation details TBD.

UPDATES:
 - 5/6 Refactoring to a better structure using OOP
 - 5/9 Right now this works fairly well in terms of both quality and compression, but the speed is completely impractical

IDEAS:
 - 5/9 Multithreading on index assignment to speed up performance

Multithreading POC script included, still in progress


