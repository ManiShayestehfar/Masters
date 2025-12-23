# 1) The big group
G := SL(2,5);

# 2) Quotient map to PSL(2,5) and identify it with A5
pi := NaturalHomomorphismByCenter(G);
A5 := Image(pi);

# 3) Pick an A4 subgroup of A5 (a point stabiliser in the natural action on 5 points)
# A5 has a natural permutation action on 5 points; a stabiliser is A4.
act := Action(A5, [1..5], OnPoints);   # permutation rep on 5 points
A5p := Image(act);                     # isomorphic copy as perm group
A4p := Stabilizer(A5p, 1);             # A4 inside that perm copy

# Pull it back to A5 (since act is an isomorphism onto A5p)
iso := IsomorphismGroups(A5, A5p);
A4  := PreImage(iso, A4p);

# 4) Pull back A4 to G; this is your H ≅ SL(2,3)
H := PreImage(pi, A4);

Size(G);  # should be 120
Size(H);  # should be 24
StructureDescription(H);  # typically reports "SL(2,3)" or "2.A4"

# 5) Choose an irrep (character) of H to induce
tH := CharacterTable(H);
irrH := Irr(tH);        # list of irreducible chars of H
chi := irrH[1];         # for example: trivial character (change index as needed)

# 6) Induce it to G and decompose
tG := CharacterTable(G);
chiG := InducedClassFunction(tH, tG, chi);

# Decompose induced character into irreducibles of G
decomp := Decomposition(tG, chiG);
irrG := Irr(tG);

decomp;  # multiplicities against irrG in the same order
