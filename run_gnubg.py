#!/usr/bin/env python3
"""simple script to run gnubg and interact with it."""

from autogamen.gnubg.interface import GnubgInterface


def run_gnubg_interactive():
    """run gnubg in interactive mode, forwarding stdin/stdout."""
    print("§ starting gnubg in interactive mode...")
    print("ℹ you can type gnubg commands directly")
    print("ℹ useful commands: help, new game, hint, show board, quit")
    print()

    with GnubgInterface() as gnubg:
        try:
            while True:
                command = input("(gnubg) ")
                if not command:
                    continue

                if command.strip() in ["quit", "exit"]:
                    break

                gnubg._send_command(command)
                output = gnubg._read_until_prompt(debug=False)

                if output:
                    print(output)

        except (KeyboardInterrupt, EOFError):
            print("\n⏹ exiting gnubg")
            return


if __name__ == "__main__":
    run_gnubg_interactive()
