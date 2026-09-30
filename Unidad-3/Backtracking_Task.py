##################### Constraint Programming and Scheduling #########

# We define CSP variables, domains and constraints

subjects = ["Artificial Intelligence",
            "Calculus",
            "Algebra",
            "Programming",
            "Statistics",
            "Physics"]

avalaible_schedules = {

    "Artificial Intelligence" :  ["08:00", "10:00", "12:00"],
    "Calculus" :  ["08:00", "10:00", "12:00"],
    "Algebra" :  ["08:00", "10:00", "12:00"],
    "Programming" :  ["08:00", "10:00", "12:00"],
    "Statistics" :  ["08:00", "10:00", "12:00"],
    "Physics" :  ["08:00", "10:00", "12:00"],

}

neighboring_subjects = {

"Artificial Intelligence" :  ["Calculus", "Programming", "Statistics"],
"Calculus" :  ["Artificial Intelligence", "Algebra", "Physics"],
"Programming" :  ["Artificial Intelligence", "Statistics"],
"Algebra" :  ["Calculus", "Physics"],
"Statistics" :  ["Artificial Intelligence", "Programming"],
"Physics" :  ["Calculus", "Algebra"]
}

############## CSP ALGORITHM ###################################

def solve_csp(variables, domains, neighbors):
    assignment = {}                              # Start with no assignments

    def backtrack():

        if len(assignment) == len(variables):    # All variables have a value
            return assignment

        for variable in variables:               # Choose an unassigned variable
            if variable not in assignment:
                break

        for value in domains[variable]:          # Try each possible value (domain, in order)
            valid = True                         # Flag to check if the assignment is valid

            for neighbor in neighbors[variable]: # Check the constraints
                if neighbor in assignment:       # If the neighbor is already assigned
                    if assignment[neighbor] == value: # If the neighbor has the same value, it's invalid
                        valid = False
                        break

            if valid == True:
                assignment[variable] = value     # Assign the value to the variable in the dictionary

                result = backtrack()             # Recursion

                if result is not None:
                    return result                # Solution found

                del assignment[variable]         # Undo assignment (backtrack)

        return None                              # No valid value
   
    result = backtrack()                         # Start the search
    return result


###################### USE OF THE ALGORITHM #################################


scheduling = solve_csp(subjects, avalaible_schedules, neighboring_subjects)

print("A valid schedule for the students is:", scheduling)



####### Questions ############
#Answer briefly:

#What are the variables in this CSP?

#Here, the variables are the subjects: Artificial Intelligence, Calculus, Algebra, programming, Statistics and Physics

#What is the domain?

# The domain of the variables are the possible schedules that the subject can take. In this case, the schedules are: 08:00, 10:00 and 12:00

#Give an example of a constraint.

# For this exercise, the constraints are distinct schedules for two variables. For example: Statistics schedule must be distinct to Programming schedule.

#Is there more than one possible valid schedule? Explain briefly.

# This algorithm only let give a first solution but if we configure the neighboring_schedules, we can get another solution.

#What happens if only two time slots are available?

