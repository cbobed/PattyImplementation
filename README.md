# Python implemementation of PATTY 
### forked from [ankitk28/CS412Project](https://github.com/ankitk28/CS412Project) 

As the name states, this project is a python implementation of PATTY (ref. to be added). 
While it started as just an adaptation of the already existing implementation to Spanish,
it has evolved in its own, suffering from a lot of refactoring and 
a lot of modifications to overcome some limitations and solve some particular problems 
that deviated the original project from the algorithm descripted in the paper. 

In order to use it, note that the current implementation has just a very simple entity type 
system (the NER tagging provided by the language model, and it's generalization to <ENTITY>). 
This is something to be extended in the future in order to improve the semantic subsumption 
capabilities. However, as we were using it to explore text corpora without having to actually 
perform a proper entity linking, it served our purposes. 

Given the implementation on top of spacy, the adaptation to any other language should be 
pretty straightforward, modifiying just the model to be loaded in the utils module and 
just extending the NE tags methods **(further doc. required here)**
to include the NEs detected by the language model you have loaded in *spacy*. 

