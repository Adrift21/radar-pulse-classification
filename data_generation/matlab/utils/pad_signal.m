function [padded, start_idx, stop_idx] = pad_signal(pulse, N, strategy)
% PAD_SIGNAL  Embed a short pulse into a fixed-length zero-padded frame.
%
% Inputs
% ------
%   pulse    : (M x 1) column vector, complex or real, M <= N
%   N        : target frame length (samples)
%   strategy : 'random' | 'left' | 'right' | 'center'
%       - 'left'   : pulse at the start, zeros after
%       - 'right'  : zeros before, pulse at the end
%       - 'center' : pulse centered, zeros symmetrically
%       - 'random' : uniform random offset within available range
%
% Outputs
% -------
%   padded    : (N x 1) frame containing the pulse
%   start_idx : 1-based index where pulse begins inside `padded`
%   stop_idx  : 1-based index where pulse ends (inclusive)
%
% Notes
% -----
% If M >= N, the pulse is truncated to the first N samples (start_idx=1).
% This should not happen with the configured pulse-width range, but the
% guard makes the function robust.

    pulse = pulse(:);                       % force column
    M = numel(pulse);

    if M >= N
        padded = pulse(1:N);
        start_idx = 1;
        stop_idx  = N;
        return;
    end

    max_offset = N - M;                     % available padding budget

    switch lower(strategy)
        case 'left'
            offset = 0;
        case 'right'
            offset = max_offset;
        case 'center'
            offset = floor(max_offset / 2);
        case 'random'
            offset = randi([0, max_offset]);
        otherwise
            error('pad_signal:unknownStrategy', ...
                  'Unknown padding strategy: %s', strategy);
    end

    padded = zeros(N, 1, 'like', pulse);    % preserve complex type
    start_idx = offset + 1;
    stop_idx  = offset + M;
    padded(start_idx : stop_idx) = pulse;
end
