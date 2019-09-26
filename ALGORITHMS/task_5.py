# Scheduling is how the processor decides which jobs(processes) get to use the processor and for how long. This can cause a lot of problems. Like a really long process taking the entire CPU and freezing all the other processes. One solution is Shortest Job First(SJF), which today you will be implementing.
#
# SJF works by, well, letting the shortest jobs take the CPU first. If the jobs are the same size then it is First In First Out (FIFO). The idea is that the shorter jobs will finish quicker, so theoretically jobs won't get frozen because of large jobs. (In practice they're frozen because of small jobs).
#
# Specification
#
# sjf(jobs, index)
#
# Returns the clock-cycles(cc) of when process will get executed for given index
#
# Parameters
# jobs: Array (of Integerss) - A non-empty array of positive integers representing cc needed to finish a job.
# index: Integer - A positive integer that respresents the job we're interested in
# Return Value
# Integer - A number representing the cc it takes to complete the job at index.
# Examples
#
# jobs	index	Return Value
# [3,10,20,1,2]	0	6
# [3,10,10,20,1,2]	2	26

import unittest


def sjf(jobs, index):
    # sorted_list = sorted(jobs)
    sorted_indxs = [i[0] for i in sorted(enumerate(jobs), key=lambda x:x[1])]
    acum = 0
    for idx in sorted_indxs:
        acum += jobs[idx]
        if index == idx:
            return acum

# TESTS


class Test(unittest.TestCase):
    def test_should_handle_the_example(self):
        self.assertEqual(sjf([3,10,20,1,2],0), 6)
        self.assertEqual(sjf([3,10,10,20,1,2],2), 26)
        self.assertEqual(sjf([10,10,10,10],3), 40)
