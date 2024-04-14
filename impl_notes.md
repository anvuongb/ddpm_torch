### Implementation notes

- Self attention block final return should be x + attns, otherwise it won't work

- Learning rate should be very small, lik 1e-5 or 1e-6, otherwise it will be nan

- Swiss or ReLU seems to be the same so far