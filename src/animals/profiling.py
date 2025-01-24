import pstats

# Load the .prof file
stats = pstats.Stats("profile_results2.prof")

# Sort and print the statistics
stats.strip_dirs()  # Remove extraneous path information
stats.sort_stats("cumtime")  # Sort by time (or 'cumulative', 'calls', etc.)
stats.print_stats(20)  # Print the top 20 time-consuming functions
