"""
Story-Focused Markov Chain Text Generator
Implements a statistical model that predicts probability of next word/character based on previous ones
Specialized for story generation with relevant training data
"""

import random
import re
from collections import defaultdict, Counter

class StoryMarkovGenerator:
    """
    Markov Chain text generator optimized for story generation
    """

    def __init__(self, state_size=2):
        """
        Initialize the generator

        Args:
            state_size (int): Number of previous words to consider (n-gram size)
        """
        self.state_size = state_size
        self.chain = defaultdict(Counter)
        self.words = []
        self.is_trained = False
        self.sentence_starters = []  # Track sentence beginnings

    def clean_text(self, text):
        """Clean and tokenize text while preserving sentence structure"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text.strip())
        # Split into sentences first
        sentences = re.split(r'[.!?]+', text)

        all_words = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                words = sentence.split()
                if words:
                    # Track sentence starters for better generation
                    if len(words) >= self.state_size:
                        self.sentence_starters.append(tuple(words[:self.state_size]))
                    all_words.extend(words)
                    all_words.append('.')  # Add period as word for sentence structure

        return all_words

    def train(self, text):
        """
        Train the Markov chain on input text

        Args:
            text (str): Training text
        """
        print("Training the model...")

        # Clean and tokenize
        self.words = self.clean_text(text)
        self.sentence_starters = []  # Reset sentence starters

        # Build the chain
        for i in range(len(self.words) - self.state_size):
            # Current state (tuple of previous words)
            current_state = tuple(self.words[i:i + self.state_size])
            # Next word
            next_word = self.words[i + self.state_size]
            # Update the chain
            self.chain[current_state][next_word] += 1

        self.is_trained = True
        print(f"✓ Model trained on {len(self.words)} words")
        print(f"✓ Created {len(self.chain)} states")
        print(f"✓ Found {len(self.sentence_starters)} sentence starters")

    def find_relevant_start(self, prompt):
        """
        Find a relevant starting state based on the prompt

        Args:
            prompt (str): User's prompt/theme

        Returns:
            tuple: Best starting state or None if not found
        """
        if not prompt:
            return None

        prompt_words = set(prompt.lower().split())

        # Look for states that contain prompt words
        best_matches = []
        for state in self.chain.keys():
            state_words = set(word.lower() for word in state)
            common_words = prompt_words.intersection(state_words)
            if common_words:
                best_matches.append((state, len(common_words)))

        if best_matches:
            # Sort by number of matching words and return the best
            best_matches.sort(key=lambda x: x[1], reverse=True)
            return best_matches[0][0]

        return None

    def generate_story(self, prompt="", length=100, use_sentence_starters=True):
        """
        Generate a story based on a prompt

        Args:
            prompt (str): Story prompt/theme
            length (int): Number of words to generate
            use_sentence_starters (bool): Use sentence starters for better flow

        Returns:
            str: Generated story
        """
        if not self.is_trained:
            return "Error: Model not trained yet! Please train the model first."

        if len(self.chain) == 0:
            return "Error: No training data available!"

        # Find relevant starting state
        current_state = self.find_relevant_start(prompt)

        if not current_state:
            # Use sentence starters if available
            if use_sentence_starters and self.sentence_starters:
                current_state = random.choice(self.sentence_starters)
            else:
                current_state = random.choice(list(self.chain.keys()))

        # Generate story
        result = list(current_state)
        sentence_word_count = len(current_state)

        for _ in range(length - self.state_size):
            if current_state in self.chain:
                # Get possible next words and their frequencies
                choices = list(self.chain[current_state].keys())
                weights = list(self.chain[current_state].values())

                # Choose next word based on probability
                next_word = random.choices(choices, weights=weights)[0]
                result.append(next_word)
                sentence_word_count += 1

                # If we hit a sentence end and have generated enough words for this sentence
                if next_word == '.' and sentence_word_count > 5:
                    # Start a new sentence
                    if self.sentence_starters and random.random() < 0.7:  # 70% chance to use sentence starter
                        current_state = random.choice(self.sentence_starters)
                        sentence_word_count = 0
                    else:
                        current_state = tuple(result[-self.state_size:])
                else:
                    # Update current state (sliding window)
                    current_state = tuple(result[-self.state_size:])
            else:
                # If current state not found, choose random state
                if self.sentence_starters and random.random() < 0.5:
                    current_state = random.choice(self.sentence_starters)
                else:
                    current_state = random.choice(list(self.chain.keys()))

        # Clean up the output
        story = ' '.join(result)
        story = re.sub(r'\s+\.', '.', story)  # Fix spacing before periods
        story = re.sub(r'\.+', '.', story)    # Fix multiple periods
        story = re.sub(r'\s+', ' ', story)    # Fix multiple spaces

        return story.strip()

    def show_statistics(self):
        """Show model statistics"""
        if not self.is_trained:
            print("Model not trained yet!")
            return

        print(f"\n📊 Model Statistics:")
        print(f"Total words in training: {len(self.words)}")
        print(f"Unique states: {len(self.chain)}")
        print(f"State size (n-gram): {self.state_size}")
        print(f"Sentence starters: {len(self.sentence_starters)}")

        # Show some example transitions
        print(f"\n🔗 Sample transitions:")
        for i, (state, next_words) in enumerate(list(self.chain.items())[:5]):
            most_common = next_words.most_common(3)
            print(f"'{' '.join(state)}' → {most_common}")

def get_story_training_data():
    """Provide diverse story training data"""
    return """
    Once upon a time, in a magical kingdom far away, there lived a brave knight named Sir Arthur. The knight rode his horse through the enchanted forest, searching for the lost princess. The princess had been captured by a fierce dragon who lived in a dark castle on top of a mountain.

    In the bustling city of New York, a young detective named Sarah was investigating a mysterious case. The case involved a stolen painting from the famous art museum. Sarah followed the clues through the busy streets, questioning witnesses and searching for evidence.

    The spaceship landed on the alien planet with a loud crash. Captain Johnson and his crew stepped out onto the strange purple soil. The alien world was filled with glowing plants and creatures they had never seen before. The crew began their mission to explore this new world and make contact with any intelligent life.

    In the old haunted mansion, strange noises echoed through the halls at midnight. The ghost of Lady Margaret walked the corridors, searching for her lost love. She had been waiting for over a hundred years, hoping that one day he would return to her.

    The young wizard Harry practiced his spells in the ancient library. Magic filled the air as he cast enchantments and created potions. His mentor, the wise old wizard Gandalf, watched proudly as his student mastered each new spell.

    Deep in the jungle, the explorer discovered a hidden temple filled with ancient treasures. Golden statues and precious gems sparkled in the torchlight. But the temple was guarded by dangerous traps and wild animals.

    The pirate ship sailed across the stormy seas, searching for buried treasure. Captain Blackbeard and his crew faced dangerous storms and rival pirates. They followed an old treasure map that led to a mysterious island.

    In the future world of 2150, robots and humans lived together in harmony. The city was filled with flying cars and towering skyscrapers. A young scientist named Dr. Elena was working on a revolutionary invention that would change the world forever.

    The cowboy rode his horse across the dusty plains of the Wild West. Sheriff John was chasing a gang of outlaws who had robbed the local bank. The chase led through desert canyons and abandoned ghost towns.

    In the underwater kingdom of Atlantis, mermaids and sea creatures lived in beautiful coral palaces. Princess Marina discovered a message in a bottle from the surface world. She decided to venture to the land above the waves to find its sender.

    The time traveler activated his machine and found himself in medieval times. He met knights, peasants, and kings in a world very different from his own. He had to be careful not to change history while trying to find his way back home.

    The vampire Count Dracula lived in his dark castle in Transylvania. He emerged at night to roam the countryside, but he was being hunted by a brave vampire hunter named Van Helsing. The hunter carried special weapons and knew all the vampire's weaknesses.

    The fairy godmother appeared to Cinderella in a shower of magical sparkles. She waved her wand and transformed the poor girl's rags into a beautiful gown. The magic would last until midnight, when Cinderella had to return from the royal ball.

    The superhero soared through the sky, protecting the city from evil villains. With incredible strength and the power of flight, he fought crime and saved innocent people. His secret identity was that of a mild-mannered reporter.

    The lost civilization was discovered deep in the Amazon rainforest. Archaeologists found ancient pyramids and mysterious artifacts. The civilization had advanced technology that was far ahead of its time.
    """

def main():
    """Main function with interactive menu"""
    print("📚 Story-Focused Markov Chain Text Generator")
    print("=" * 55)

    # Initialize generator
    generator = StoryMarkovGenerator(state_size=2)

    # Auto-train with story data
    print("\n📝 Auto-training with story data...")
    story_data = get_story_training_data()
    generator.train(story_data)
    generator.show_statistics()

    # Generate some automatic examples
    print("\n🎲 Random Story Generations:")
    sample_prompts = ["fantasy adventure", "mystery detective", "space exploration", "haunted house", "magical kingdom"]

    for i, prompt in enumerate(sample_prompts[:3]):
        generated = generator.generate_story(prompt=prompt, length=40)
        print(f"{i+1}. Prompt: '{prompt}'")
        print(f"   Story: {generated}\n")

    # Interactive menu
    while True:
        print("\n" + "="*55)
        print("📋 Options:")
        print("1. Generate story with prompt")
        print("2. Generate random story")
        print("3. Train on new story text")
        print("4. Show model statistics")
        print("5. Change state size")
        print("6. Exit")

        try:
            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == '1':
                # Generate story with prompt
                prompt = input("Enter your story prompt (e.g., 'fantasy adventure', 'mystery detective'): ").strip()
                length = input("Enter story length (default 80): ").strip()
                length = int(length) if length.isdigit() else 80

                generated = generator.generate_story(prompt=prompt, length=length)
                print(f"\n📖 Generated Story:")
                print(f"Prompt: '{prompt}'")
                print(f"Story: {generated}")

            elif choice == '2':
                # Generate random story
                length = input("Enter story length (default 80): ").strip()
                length = int(length) if length.isdigit() else 80

                generated = generator.generate_story(length=length)
                print(f"\n📖 Generated Story:\n{generated}")

            elif choice == '3':
                # Train on new text
                print("\nEnter your story text (press Enter twice to finish):")
                lines = []
                empty_line_count = 0
                while True:
                    line = input()
                    if line == "":
                        empty_line_count += 1
                        if empty_line_count >= 2:
                            break
                    else:
                        empty_line_count = 0
                    lines.append(line)

                text = '\n'.join(lines)
                if text.strip():
                    generator.train(text)
                    generator.show_statistics()
                else:
                    print("✗ No text entered!")

            elif choice == '4':
                # Show statistics
                generator.show_statistics()

            elif choice == '5':
                # Change state size
                new_state_size = input(f"Enter new state size (current: {generator.state_size}): ").strip()
                if new_state_size.isdigit():
                    generator.state_size = int(new_state_size)
                    generator.chain = defaultdict(Counter)
                    generator.is_trained = False
                    print("✓ State size updated! Please retrain the model.")
                else:
                    print("✗ Invalid state size!")

            elif choice == '6':
                print("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice! Please try again.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
