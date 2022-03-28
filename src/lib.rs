#[macro_use(s)]
#[cfg_attr(test, macro_use(array))]
extern crate ndarray;

use std::fmt;

pub mod duplex;
pub mod search;
pub mod tree;
pub mod vec2d;

#[cfg(feature = "fastexp")]
mod fastexp;

pub mod prelude {
    pub use super::duplex::*;
    pub use super::search::*;
    pub use super::tree::*;
    pub use super::vec2d::*;
}

#[derive(Clone, Copy, Debug)]
pub enum SearchError {
    RanOutOfBeam,
    IncomparableValues,
    InvalidEnvelope,
}

impl fmt::Display for SearchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchError::RanOutOfBeam => {
                write!(f, "Ran out of search space (beam_cut_threshold too high)")
            }
            SearchError::IncomparableValues => {
                write!(f, "Failed to compare values (NaNs in input?)")
            }
            // TODO: document envelope constraints
            SearchError::InvalidEnvelope => write!(f, "Invalid envelope values"),
        }
    }
}

impl std::error::Error for SearchError {}


// / Perform a CTC beam search decode on an RNN output.
// /
// / This function does a beam search variant of the prefix search decoding mentioned (and described
// / in fairly vague terms) in the original CTC paper (Graves et al, 2006, section 3.2).
// /
// / The paper mentioned above provides recursive equations that give an efficient way to find the
// / probability for a specific labelling. A tree of possible labelling suffixes, together with
// / their probabilities, can be built up by starting at one end and trying every possible label at
// / each stage. The "beam" part of the search is how we keep the search space managable - at each
// / step, we ignore all but the most-probable tree leaves (like searching with a torch beam). This
// / means we may not actually find the most likely labelling, but it often works very well.
// /
// / See the module-level documentation for general requirements on `network_output` and `alphabet`.
// /
// / Args:
// /     network_output (numpy.ndarray): The 2D array output of the neural network.
// /     alphabet (sequence): The labels (including the blank label, which must be first) in the
// /         order given on the inner axis of `network_output`.
// /     beam_size (int): How many search points should be kept at each step. Higher numbers are
// /         less likely to discard the true labelling, but also make it slower and more memory
// /         intensive. Must be at least 1.
// /     beam_cut_threshold (float): Ignore any entries in `network_output` below this value. Must
// /         be at least 0.0, and less than ``1/len(alphabet)``.
// /
// / Returns:
// /     tuple of (str, numpy.ndarray): The decoded sequence and an array of the final
// /         timepoints of each label (as indices into the outer axis of `network_output`).
// /
// / Raises:
// /     PyValueError: The constraints on the arguments have not been met.


// / Perform a CTC beam search decode on two RNN outputs that describe the same sequence.
// /
// / This is a variation of `beam_search` that attempts to find a common labelling for two RNN
// / outputs. This could be the same network run over two different samplings of the same sequence,
// / or two different networks run over the same input, for example.
// /
// / It is an implementation of the algorithm developed by Silvestre-Ryan and Holmes
// / (https://doi.org/10.1101/2020.02.25.956771).
// /
// / If no envelope is provided, a default one will be used. For now, that will just search the
// / whole of `network_output_2` at every step, but in future it may calculate a more constrained
// / envelope. For consistent results, you should provide an envelope.
// /
// / Args:
// /     network_output_1 (numpy.ndarray): The 2D array output of the first neural network.
// /     network_output_2 (numpy.ndarray): The 2D array output of the second neural network. Note
// /         that while the inner axis size must match that of `network_output_1`, the outer axis
// /         can be a different size.
// /     alphabet (str): The labels (including the blank label, which must be first) in the order
// /         given on the inner axis of `network_output_1` and `network_output_2`.
// /     envelope (numpy.ndarray, optional): An Nx2 array, where N is the outer axis length of
// /         `network_output_1`. For each row of `network_output_1`, this gives the starting and
// /         ending rows of `network_output_2` to consider for alignment.
// /     beam_size (int): How many suffix_tree should be kept at each step. Higher numbers are less
// /         likely to discard the true labelling, but also make it slower and more memory
// /         intensive. Must be at least 1.
// /     beam_cut_threshold (float): Ignore any entries in `network_output` below this value. Must
// /         be at least 0.0, and less than ``1/len(alphabet)``.
// /
// / Returns:
// /     str: The decoded sequence.
// /
// / Raises:
// /     PyValueError: The constraints on the arguments have not been met.



// / Methods for labelling RNN results using CTC decoding.
// /
// / The methods in this module implement the last step of labelling input data. In the case of
// / nanopore sequencing data, we're taking the electrical current samples, and labelling them with
// / what we think the DNA/RNA base is at any given time.
// /
// / CTC decoding (and hence the funtions in this module) takes as input the result of a neural
// / network that has figured out, for each sample and each label, the probability that the sample
// / corresponds to that label. The network also outputs a probability for the data point
// / corresponding to an extra "blank" label (a sort of "none of the above" option). This is
// / represented as a 2D matrix of size ``N x (L+1)``, where ``N`` is the number of samples and
// / ``L`` is the number of labels we're interested in (the ``+1`` is to account for the blank
// / label).
// /
// / A _path_ through the matrix is an assignment of a label or blank to each sample. The
// / probability that the path is correct is the product of the selected entries in the matrix. Each
// / path produces a labelling: first collapse all duplicate labels or blanks, then remove the
// / remaining blanks - AAAGGbGGbbbC would become AGbGbC, and then AGGC. The probability that the
// / labelling is correct is the sum of the probabilities of the paths that produce it. We want the
// / most likely labelling.
// /
// / This problem is, in general, intractable. This module provides heuristic functions that attempt
// / to find the most likely labelling (but may produce a suboptimal labelling).
// /
// / All functions take outputs from one or more neural networks, plus an alphabet to use for the
// / labelling.
// /
// / The network outpus are 2D arrays produced by a softmax layer of a neural network, with values
// / between 0.0 and 1.0 representing probabilities. The outer axis (rows) is time, and the inner
// / axis (columns) is labels. The first entry on the label axis is assumed to be the blank label.
// / It's also worth noting that the values in each row should sum to 1.0.
// /
// / The alphabet can be a str or any sequence of str (eg: a list or tuple of str). Each element (or
// / character in the case of str) provides the labelling for one element of the inner axis of the
// / network output(s) - therefore, len(alphabet) must be the size of that inner axis. Using a list
// / or tuple allows multi-character labels to be specified. Note that the first label is not
// / actually used by any of the functions in this module, so the value does not matter.
