import numpy as np
from collections import defaultdict

class RobotLanguageModel:
    def __init__(self, sequence):
        """
        Initialize the robot language model with a movement sequence.
        
        Args:
            sequence (str): The movement sequence (C=Start, A=Advance, D=Right, I=Left)
        """
        self.sequence = sequence
        self.movements = list(sequence)
        
    def calculate_probabilities(self, n_orders):
        """
        Calculate conditional probabilities P(Xi|Xi-1,..,Xi-n)
        
        Args:
            n_orders (int): Number of previous orders to remember
            
        Returns:
            dict: Dictionary with context as key and probability distribution as value
        """
        probabilities = defaultdict(lambda: defaultdict(int))
        context_counts = defaultdict(int)
        
        # Count occurrences of contexts and next movements
        for i in range(n_orders, len(self.movements)):
            context = tuple(self.movements[i-n_orders:i])
            next_movement = self.movements[i]
            
            probabilities[context][next_movement] += 1
            context_counts[context] += 1
        
        # Convert counts to probabilities
        for context in probabilities:
            total = context_counts[context]
            for movement in probabilities[context]:
                probabilities[context][movement] /= total
                
        return probabilities, context_counts
    
    def generate_movement(self, context, probabilities):
        """
        Generate the next movement based on the most probable option.
        
        Args:
            context (tuple): Current context (previous n movements)
            probabilities (dict): Probability distributions
            
        Returns:
            str: Next movement or None if context not found
        """
        if context not in probabilities:
            # Print error and return None to indicate failure
            print(f"❌ ERROR: Contexto '{' '.join(context)}' NO ENCONTRADO en las probabilidades")
            print(f"   Contextos disponibles: {list(probabilities.keys())}")
            return None
        
        # Return movement with highest probability
        return max(probabilities[context].items(), key=lambda x: x[1])[0]
    
    def simulate_path(self, n_orders, max_steps=100):
        """
        Simulate the robot's path using the language model.
        
        Args:
            n_orders (int): Number of previous orders to remember
            max_steps (int): Maximum number of steps to prevent infinite loops
            
        Returns:
            list: List of movements generated
        """
        probabilities, _ = self.calculate_probabilities(n_orders)
        generated_movements = []
        
        # Start with the first n movements from the original sequence
        current_context = tuple(self.movements[:n_orders])
        
        for step in range(max_steps):
            next_movement = self.generate_movement(current_context, probabilities)
            
            # If context not found, stop the simulation
            if next_movement is None:
                print(f"🛑 SIMULACIÓN DETENIDA: No se puede continuar sin probabilidades para el contexto")
                break
                
            generated_movements.append(next_movement)
            
            # Update context
            current_context = current_context[1:] + (next_movement,)
            
            # Check if we've reached a stopping condition
            if len(generated_movements) >= len(self.sequence):
                # Check if we've repeated the original sequence
                if generated_movements[-len(self.sequence):] == self.movements:
                    break
                    
        return generated_movements

def test_robot_algorithm():
    """
    Simple test function to test n=1 to n=5.
    """
    # Original sequence
    sequence = "CAAAIADAIAAIAADAADAIAAIAA"
    robot = RobotLanguageModel(sequence)
    
    print("=== ANÁLISIS SIMPLE DEL ROBOT ===\n")
    print(f"Secuencia original: {sequence}")
    print(f"Longitud: {len(sequence)} movimientos\n")
    
    # Test n=1 to n=5
    for n in range(1, 6):
        print(f"--- PRUEBA CON n={n} ---")
        
        # 1. Calculate probabilities
        probabilities, counts = robot.calculate_probabilities(n)
        print(f"Contextos encontrados: {len(probabilities)}")
        
        # Show ALL probabilities for this n
        print("Probabilidades:")
        for context, probs in probabilities.items():
            context_str = ''.join(context)
            print(f"  P(X|{context_str}) = {dict(probs)}")
        
        # 2. Generate sequence using max probability
        print(f"\nGenerando secuencia con n={n}:")
        generated_path = robot.simulate_path(n, max_steps=50)
        
        # 3. Check results
        print(f"Movimientos generados: {len(generated_path)}")
        if len(generated_path) > 0:
            print(f"Secuencia generada: {''.join(generated_path[:30])}...")
            
            # Check if it matches original
            if len(generated_path) >= len(sequence):
                if generated_path[:len(sequence)] == robot.movements:
                    print("✅ ÉXITO: La secuencia generada coincide con la original!")
                else:
                    print("❌ La secuencia generada NO coincide con la original")
            else:
                print("❌ No se generaron suficientes movimientos")
        else:
            print("❌ No se pudo generar ninguna secuencia")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_robot_algorithm()
