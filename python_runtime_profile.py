import pstats
p =  pstats.Stats("profile.out")
p.strip_dirs().sort_stats("cumtime").print_stats(40)