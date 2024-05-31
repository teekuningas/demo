{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.11";
    flake-utils.url = "github:numtide/flake-utils";

    nixgl.url = "github:guibou/nixGL";
    nixgl.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, nixgl, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            qt5.full
            (pkgs.python39.withPackages (ps: [ ps.matplotlib ps.vispy ps.numpy ps.black ps.pyqt5 ]))
          ];
        };
        packages.nixgl = nixgl.packages.${system}.nixGLIntel;
      });
}
