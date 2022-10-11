import json


def write_test_results_to_file(results_in, fname):
    """Dump partial WFS&C loop results into txt file.

    This function is useful when you have results from a new benchmark regression test that you would like to
    write new tests for. Saves the items 'nb_total_iter', 'Nb_iter_per_mat' and 'MeanDHContrast'.

    Parameters
    ----------
    results_in : dict
        A dictionary result returned by Asterix.main_THD.runthd2().
    fname : string
        Full path and filename, including ".txt" at the end, for the file to be written out.
    """
    results = {
        'nb_total_iter': results_in['nb_total_iter'],
        'Nb_iter_per_mat': results_in['Nb_iter_per_mat'],
        'MeanDHContrast': results_in['MeanDHContrast']
    }

    with open(fname, 'w') as convert_file:
        convert_file.write(json.dumps(results))


def read_test_results_from_file(fname):
    """Read regression test results from file.

    Read test results 'nb_total_iter', 'Nb_iter_per_mat', 'MeanDHContrast' from a file they have previously been saved
    to with write_test_results_to_file(). Used for writing new regression tests for the WFS&C loops.

    Parameters
    ----------
    fname : string
        Absolute path and filename to file being read, including ".txt" at the end.

    Returns
    -------
    results : dict
        A slimmed down dictionarry as returned by Asterix.main_THD.runthd2(),containing only the items with keys
        'nb_total_iter', 'Nb_iter_per_mat' and 'MeanDHContrast'.
    """
    with open(fname, 'r') as f:
        results = json.load(f)

    return results
