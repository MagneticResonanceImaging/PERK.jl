"""
    div0(a, b)

Compute `a / b`, but return 0 if `b` is 0.
"""
function div0(dividend::Number, divisor::Number)

    tmp = dividend / divisor
    return divisor == 0 ? zero(tmp) : tmp

end
