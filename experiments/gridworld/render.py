def render_value_function(env, v):
    """Prints value function in 2d grid"""
    out = "-"*(env.Nx*10+1) + "\n"
    for y in range(env.Ny):
        for x in range(env.Nx):
            s = y*env.Nx + x
            out += "| {:>7.2f} ".format(v[s])
        out += "|\n"
    out += "-"*(env.Nx*10+1) + "\n"
    print(out)


def render_policy_and_value_function(env, pi, v):
    """Prints policy and value function in 2d grid"""
    out = "-"*(env.Nx*10+1) + "     " + "-"*(env.Nx*7+1) + "\n"
    for y in range(env.Ny):
        for x in range(env.Nx):
            s = y*env.Nx + x
            out += "| {:>7.2f} ".format(v[s])

        out += "|     "
        for x in range(env.Nx):
            out += "| "
            s = y*env.Nx + x
            if s in env.terminal_states:
                out += "  T  "
                continue
            if pi[s][3] != 0:
                out += "<"
            else:
                out += " "
            if pi[s][0] != 0:
                out += "v"
            else:
                out += " "
            if pi[s][1] != 0:
                out += "^"
            else:
                out += " "
            if pi[s][2] != 0:
                out += ">"
            else:
                out += " "
            out += " "
        out += "|\n"
    out += "-"*(env.Nx*10+1) + "     " + "-"*(env.Nx*7+1) + "\n"
    print(out)
