**Placeholder Directory for Original Data Files**    

Data can be downloaded from: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html    

Please download the following and place them in this directory:  
Detailed book graph (~2gb, about 2.3m books): **goodreads_books.json.gz**  
Detailed information of authors: **goodreads_book_authors.json.gz**  
Extracted fuzzy book genres (genre tags are extracted from users' popular shelves by a simple keyword matching process): **goodreads_book_genres_initial.json.gz**  
Complete user-book interactions in 'csv' format (~4.1gb): **goodreads_interactions.csv**    
User Ids and Book Ids in this file can be reconstructed by joining on the following two files: **book_id_map.csv, user_id_map.csv.**    
Detailed information of the complete user-book interactions (~11gb, ~229m records): **goodreads_interactions_dedup.json.gz**    
Complete book reviews (~15m multilingual reviews about ~2m books and 465k users): **goodreads_reviews_dedup.json.gz**   


Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.    
Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19. 
