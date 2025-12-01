"""
Quick setup script for AI Akinator project.

This script helps you set up the project quickly by:
1. Creating virtual environment
2. Installing dependencies
3. Setting up .env file
4. Validating configuration
"""

import subprocess
import sys
import os
from pathlib import Path
import shutil


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")


def run_command(cmd, cwd=None, shell=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=shell,
            check=True,
            capture_output=True,
            text=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def setup_virtual_environment():
    """Create and set up virtual environment."""
    print_header("🔧 Setting Up Virtual Environment")
    
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            print("🗑️  Removing old virtual environment...")
            shutil.rmtree(venv_path)
        else:
            print("✅ Using existing virtual environment")
            return True
    
    print("📦 Creating virtual environment...")
    success, output = run_command([sys.executable, "-m", "venv", "venv"])
    
    if success:
        print("✅ Virtual environment created successfully!")
        return True
    else:
        print(f"❌ Failed to create virtual environment: {output}")
        return False


def install_dependencies():
    """Install project dependencies."""
    print_header("📚 Installing Dependencies")
    
    # Determine pip path
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip.exe")
    else:
        pip_path = Path("venv/bin/pip")
    
    if not pip_path.exists():
        print(f"❌ pip not found at {pip_path}")
        return False
    
    print("📥 Installing packages from requirements.txt...")
    success, output = run_command([str(pip_path), "install", "-r", "requirements.txt"])
    
    if success:
        print("✅ Dependencies installed successfully!")
        print("\n📋 Installed packages:")
        run_command([str(pip_path), "list"])
        return True
    else:
        print(f"❌ Failed to install dependencies: {output}")
        return False


def setup_env_file():
    """Set up .env file from template."""
    print_header("🔐 Setting Up Environment Variables")
    
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    if env_path.exists():
        print("⚠️  .env file already exists")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("✅ Keeping existing .env file")
            return True
    
    if not env_example_path.exists():
        print("❌ .env.example file not found")
        return False
    
    # Copy template
    shutil.copy(env_example_path, env_path)
    print("✅ Created .env file from template")
    
    print("\n⚠️  IMPORTANT: You need to add your API keys to the .env file!")
    print("\n📝 Required API keys:")
    print("   1. ANTHROPIC_API_KEY - Get from: https://console.anthropic.com")
    print("   2. LANGCHAIN_API_KEY - Get from: https://smith.langchain.com")
    
    response = input("\nDo you want to enter your API keys now? (y/N): ").lower()
    
    if response == 'y':
        anthropic_key = input("Enter ANTHROPIC_API_KEY: ").strip()
        langsmith_key = input("Enter LANGCHAIN_API_KEY: ").strip()
        
        # Read .env content
        with open(env_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace('your_anthropic_key_here', anthropic_key)
        content = content.replace('your_langsmith_key_here', langsmith_key)
        
        # Write back
        with open(env_path, 'w') as f:
            f.write(content)
        
        print("✅ API keys added to .env file")
    else:
        print("\n💡 Don't forget to edit .env file and add your API keys later!")
    
    return True


def validate_setup():
    """Validate the setup."""
    print_header("✅ Validating Setup")
    
    # Determine python path
    if sys.platform == "win32":
        python_path = Path("venv/Scripts/python.exe")
    else:
        python_path = Path("venv/bin/python")
    
    if not python_path.exists():
        print(f"❌ Python not found at {python_path}")
        return False
    
    print("🧪 Testing imports...")
    test_script = """
import sys
try:
    import langgraph
    import langchain
    import langsmith
    from dotenv import load_dotenv
    print("✅ All core dependencies imported successfully")
    sys.exit(0)
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
"""
    
    success, output = run_command([str(python_path), "-c", test_script])
    
    if success:
        print(output)
        print("\n🎉 Setup validation passed!")
        return True
    else:
        print(f"❌ Validation failed: {output}")
        return False


def print_next_steps():
    """Print next steps for the user."""
    print_header("🚀 Next Steps")
    
    print("Phase 1 is now complete! Here's what to do next:\n")
    
    print("1. Activate the virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Make sure your .env file has valid API keys")
    print("   - Edit .env file")
    print("   - Add ANTHROPIC_API_KEY from https://console.anthropic.com")
    print("   - Add LANGCHAIN_API_KEY from https://smith.langchain.com")
    
    print("\n3. Test the configuration:")
    print("   python src/config.py")
    
    print("\n4. Ready for Phase 2!")
    print("   Phase 2 will implement state management and data models")
    
    print("\n📚 Documentation:")
    print("   - README.md - Project overview")
    print("   - PHASE1_CHECKLIST.md - Phase 1 completion checklist")
    print("   - Akinator_Implementation_Plan.md - Full implementation roadmap")
    
    print("\n✨ Happy coding! 🧞")


def main():
    """Main setup function."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║              🧞 AI AKINATOR - SETUP SCRIPT 🧞              ║
    ║                                                           ║
    ║            Phase 1: Project Setup & Foundation            ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("This script will help you set up the AI Akinator project.\n")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ Error: requirements.txt not found")
        print("💡 Make sure you're in the akinator-ai directory")
        sys.exit(1)
    
    # Run setup steps
    steps = [
        ("Virtual Environment", setup_virtual_environment),
        ("Dependencies", install_dependencies),
        ("Environment File", setup_env_file),
        ("Validation", validate_setup),
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Setup failed at step: {step_name}")
            print("💡 Please fix the error and run the script again")
            sys.exit(1)
    
    # Print success and next steps
    print_next_steps()


if __name__ == "__main__":
    main()
