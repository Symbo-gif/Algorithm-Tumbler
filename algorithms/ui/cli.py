import argparse
from ..library.manager import ComponentLibrary
from ..evolution.search import EvolutionarySearch

def main():
    parser = argparse.ArgumentParser(description="Algorithm Tumbler CLI")
    subparsers = parser.add_subparsers(dest='command')

    # init
    subparsers.add_parser('init', help='Initialize library from seed algorithms')

    # spin
    spin_parser = subparsers.add_parser('spin', help='Random spin')
    spin_parser.add_argument('--mode', default='random')
    spin_parser.add_argument('--steps', type=int, default=100)

    # evolve
    evolve_parser = subparsers.add_parser('evolve', help='Guided evolution')
    evolve_parser.add_argument('--objective', required=True)
    evolve_parser.add_argument('--generations', type=int, default=50)

    # add-algorithm
    add_parser = subparsers.add_parser('add-algorithm', help='Add custom algorithm')
    add_parser.add_argument('--file', required=True)

    # inspect
    inspect_parser = subparsers.add_parser('inspect', help='Inspect library')
    inspect_parser.add_argument('--top-k', type=int, default=10)

    args = parser.parse_args()

    if args.command == 'init':
        library = ComponentLibrary()
        # Load seed algorithms (placeholder)
        library.save('library.json')
        print("Library initialized")

    elif args.command == 'spin':
        library = ComponentLibrary.load('library.json')
        search = EvolutionarySearch(library, seed=42)
        search.initialize(10)
        results = []
        for _ in range(args.steps):
            # Simplified spin
            results.append(search.population[0])
        print(f"Spun {len(results)} programs")

    elif args.command == 'evolve':
        library = ComponentLibrary.load('library.json')
        search = EvolutionarySearch(library, seed=42)
        search.initialize(10)
        test_cases = []  # Placeholder
        search.evolve(args.generations, lambda p, tc: 0.0, test_cases, mode="guided")
        print("Evolution complete")

    elif args.command == 'add-algorithm':
        # Placeholder
        print(f"Added algorithm from {args.file}")

    elif args.command == 'inspect':
        library = ComponentLibrary.load('library.json')
        for i, (cid, comp) in enumerate(list(library.components.items())[:args.top_k]):
            print(f"{i+1}. {cid}: {comp}")

if __name__ == '__main__':
    main()
